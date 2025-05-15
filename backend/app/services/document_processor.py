import os
import re
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Union
import uuid
import concurrent.futures
import functools
import time
import pickle
import hashlib
import threading
import queue
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('document_processor')

# Advanced persistent cache with disk storage
class PersistentCache:
    def __init__(self, cache_dir="/tmp/doc_processor_cache", max_memory_items=100):
        self.memory_cache = {}
        self.cache_dir = cache_dir
        self.max_memory_items = max_memory_items
        self.access_times = {}
        self.lock = threading.RLock()
        
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_key(self, key_data):
        if isinstance(key_data, str):
            return hashlib.md5(key_data.encode()).hexdigest()
        return hashlib.md5(str(key_data).encode()).hexdigest()

    def _get_cache_path(self, key):
        hashed_key = self._get_cache_key(key)
        return os.path.join(self.cache_dir, f"{hashed_key}.cache")

    def get(self, key, default=None):
        with self.lock:
            if key in self.memory_cache:
                self.access_times[key] = time.time()
                logger.info(f"Memory cache hit for {key}")
                return self.memory_cache[key]
            
            cache_path = self._get_cache_path(key)
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        value = pickle.load(f)
                    
                    self._add_to_memory_cache(key, value)
                    logger.info(f"Disk cache hit for {key}")
                    return value
                except Exception as e:
                    logger.error(f"Error loading from disk cache: {e}")
        
        return default

    def set(self, key, value):
        with self.lock:
            self._add_to_memory_cache(key, value)
            cache_path = self._get_cache_path(key)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                logger.error(f"Error saving to disk cache: {e}")

    def _add_to_memory_cache(self, key, value):
        self.memory_cache[key] = value
        self.access_times[key] = time.time()
        
        if len(self.memory_cache) > self.max_memory_items:
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.memory_cache[lru_key]
            del self.access_times[lru_key]

# Improved memoize decorator with persistent caching
def persistent_memoize(func):
    cache = PersistentCache()
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key from function name and arguments
        key = f"{func.__name__}_{args}_{kwargs}"
        result = cache.get(key)
        if result is None:
            result = func(*args, **kwargs)
            cache.set(key, result)
        return result
    return wrapper

# Background processing queue
class ProcessingQueue:
    def __init__(self, max_size=100):
        self.queue = queue.Queue(maxsize=max_size)
        self.results = {}
        self.lock = threading.RLock()
        self.processing_thread = None
        self.running = False
    
    def start(self, processor_func):
        """Start the background processing thread"""
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._process_queue, 
            args=(processor_func,),
            daemon=True
        )
        self.processing_thread.start()
    
    def stop(self):
        """Stop the background processing thread"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
    
    def _process_queue(self, processor_func):
        """Process items from the queue in background"""
        while self.running:
            try:
                # Get item from queue with timeout to allow checking running flag
                item_id, item = self.queue.get(timeout=0.5)
                try:
                    # Process the item
                    result = processor_func(item)
                    # Store the result
                    with self.lock:
                        self.results[item_id] = {
                            'status': 'completed',
                            'result': result
                        }
                except Exception as e:
                    # Store the error
                    with self.lock:
                        self.results[item_id] = {
                            'status': 'error',
                            'error': str(e)
                        }
                finally:
                    self.queue.task_done()
            except queue.Empty:
                # Queue is empty, just continue
                pass
    
    def add_item(self, item_id, item):
        """Add an item to the processing queue"""
        with self.lock:
            # If already in results, return immediately
            if item_id in self.results:
                return
            # Mark as queued
            self.results[item_id] = {'status': 'queued'}
        
        # Add to queue
        self.queue.put((item_id, item))
    
    def get_result(self, item_id):
        """Get the processing result for an item"""
        with self.lock:
            return self.results.get(item_id, {'status': 'unknown'})

class DocumentProcessor:
    """Handles document processing, OCR, and answer extraction with advanced optimizations"""
    
    def __init__(self, max_workers=8, cache_dir="/tmp/doc_processor_cache"):
        # Configure Google Gemini API if not already configured
        if not os.getenv("GOOGLE_API_KEY"):
            from dotenv import load_dotenv
            load_dotenv()
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
        
        self.model = genai.GenerativeModel('gemma-3-1b-it')
        self.max_workers = max_workers
        
        # Initialize persistent cache
        self.cache = PersistentCache(cache_dir=cache_dir, max_memory_items=200)
        
        # Initialize thread pool executor for parallel processing
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Initialize background processing queue
        self.processing_queue = ProcessingQueue()
        self.processing_queue.start(self._process_document_internal)
        
        # Preload OCR engines to avoid initialization delay
        self._preload_ocr_engines()
    
    def _preload_ocr_engines(self):
        """Preload OCR engines in background to avoid initialization delay"""
        def preload():
            try:
                # Preload PaddleOCR
                try:
                    from paddleocr import PaddleOCR
                    self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
                    logger.info("PaddleOCR preloaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to preload PaddleOCR: {e}")
                    self.paddle_ocr = None
                
                # Preload Tesseract (just make a simple call to ensure it's initialized)
                try:
                    sample_image = Image.new('RGB', (100, 30), color=(255, 255, 255))
                    pytesseract.image_to_string(sample_image)
                    logger.info("Tesseract OCR preloaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to preload Tesseract: {e}")
            except Exception as e:
                logger.error(f"Error in OCR preloading: {e}")
        
        # Start preloading in background
        threading.Thread(target=preload, daemon=True).start()
    
    def process_document(self, file_path):
        """Process a document and extract its text content with instant response"""
        # Generate a unique ID for this document
        doc_id = hashlib.md5(file_path.encode()).hexdigest()
        
        # Check persistent cache first
        cached_result = self.cache.get(file_path)
        if cached_result is not None:
            logger.info(f"Cache hit for {file_path}")
            return cached_result
        
        # Check if already in processing queue
        queue_result = self.processing_queue.get_result(doc_id)
        
        if queue_result['status'] == 'completed':
            # Already processed, return result
            result = queue_result['result']
            # Also update cache
            self.cache.set(file_path, result)
            return result
        elif queue_result['status'] == 'error':
            # Processing failed
            return f"Error processing document: {queue_result.get('error', 'Unknown error')}"
        elif queue_result['status'] == 'queued':
            # Still processing, return placeholder
            return "Document is being processed. Please try again in a moment."
        
        # Start processing in background and return immediately
        self.processing_queue.add_item(doc_id, file_path)
        
        # Try to process synchronously with a short timeout for better user experience
        try:
            # Process with a very short timeout to see if it can complete quickly
            result = self._process_document_with_timeout(file_path, timeout=0.5)
            if result:
                # Processing completed quickly, update cache and return
                self.cache.set(file_path, result)
                return result
        except concurrent.futures.TimeoutError:
            # Processing taking too long, will continue in background
            pass
        
        return "Document processing started. Please try again in a moment."
    
    def _process_document_with_timeout(self, file_path, timeout=0.5):
        """Try to process document with a timeout"""
        future = self.executor.submit(self._process_document_internal, file_path)
        return future.result(timeout=timeout)
    
    def _process_document_internal(self, file_path):
        """Internal method to process document without caching logic"""
        start_time = time.time()
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Process based on file type
        try:
            if file_extension in [".jpg", ".jpeg", ".png"]:
                result = self._process_image(file_path)
            elif file_extension == ".pdf":
                result = self._process_pdf(file_path)
            elif file_extension == ".txt":
                result = self._process_text(file_path)
            else:
                result = "Unsupported file format"
            
            # Update cache
            self.cache.set(file_path, result)
            
            end_time = time.time()
            logger.info(f"Document processing took {end_time - start_time:.2f} seconds")
            
            return result
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return f"Error processing document: {str(e)}"
    
    @persistent_memoize
    def _process_image(self, image_path: str) -> str:
        """Extract text from an image using OCR with optimizations"""
        try:
            # Use preloaded PaddleOCR if available
            if hasattr(self, 'paddle_ocr') and self.paddle_ocr is not None:
                try:
                    result = self.paddle_ocr.ocr(image_path, cls=True)
                    
                    # Extract text from result
                    text = ""
                    if result and result[0]:
                        for line in result[0]:
                            text += line[1][0] + "\n"
                    
                    if text.strip():
                        return text
                except Exception as e:
                    logger.warning(f"Preloaded PaddleOCR failed, trying fresh instance: {e}")
            
            # Try PaddleOCR with fresh instance if preloaded version failed
            try:
                from paddleocr import PaddleOCR
                ocr = PaddleOCR(use_angle_cls=True, lang='en')
                result = ocr.ocr(image_path, cls=True)
                
                # Extract text from result
                text = ""
                if result and result[0]:
                    for line in result[0]:
                        text += line[1][0] + "\n"
                
                if text.strip():
                    return text
            except Exception as e:
                logger.warning(f"PaddleOCR failed, falling back to Tesseract: {e}")
            
            # Fall back to Tesseract OCR with optimizations
            image = Image.open(image_path)
            
            # Optimize image for OCR
            # 1. Convert to grayscale
            image = image.convert('L')
            
            # 2. Apply some basic image enhancements
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)  # Increase contrast
            
            # 3. Use optimized Tesseract configuration
            custom_config = r'--oem 3 --psm 6 -l eng'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            return text
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return f"Error processing image: {str(e)}"
    
    @persistent_memoize
    def _process_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file using highly optimized parallel processing"""
        try:
            start_time = time.time()
            
            # Generate a unique cache key for this PDF
            pdf_hash = hashlib.md5(open(pdf_path, 'rb').read(8192)).hexdigest()  # Hash first 8KB for speed
            cache_dir = f"/tmp/pdf_cache_{pdf_hash}"
            os.makedirs(cache_dir, exist_ok=True)
            
            # Check if we have a cached result for the entire PDF
            cache_file = os.path.join(cache_dir, "full_text.txt")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    logger.info(f"Using cached PDF result for {pdf_path}")
                    return f.read()
            
            # Optimize PDF conversion parameters
            convert_params = {
                'thread_count': self.max_workers,
                'use_pdftocairo': True,  # Usually faster than pdftoppm
                'grayscale': True,  # Faster processing, sufficient for OCR
                'size': (1000, None),  # Resize to reasonable resolution
                'fmt': 'jpeg',  # JPEG is faster than PNG for this purpose
                'jpegopt': {'quality': 90, 'optimize': True}
            }
            
            # Convert PDF to images with optimized parameters
            images = convert_from_path(pdf_path, **convert_params)
            convert_time = time.time()
            logger.info(f"PDF conversion took {convert_time - start_time:.2f} seconds")
            
            # Process pages in parallel with optimized memory usage
            futures = []
            
            for i, image in enumerate(images):
                # Check if we have a cached result for this page
                page_cache = os.path.join(cache_dir, f"page_{i}.txt")
                if os.path.exists(page_cache):
                    with open(page_cache, 'r', encoding='utf-8') as f:
                        page_text = f.read()
                    futures.append((i, page_text))
                    continue
                
                # Save image to memory buffer instead of disk
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='JPEG', quality=90, optimize=True)
                img_buffer.seek(0)
                
                # Submit image processing task
                future = self.executor.submit(self._process_image_buffer, img_buffer, i, cache_dir)
                futures.append((i, future))
            
            # Collect results in order
            results = [None] * len(images)
            for i, future in futures:
                if isinstance(future, str):
                    # This is a cached result
                    results[i] = (i, future)
                else:
                    # This is a Future object
                    results[i] = (i, future.result())
            
            # Combine results in page order
            full_text = ""
            for i, page_text in sorted(results):
                full_text += f"\n--- Page {i+1} ---\n{page_text}"
            
            # Cache the full result
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
            
            end_time = time.time()
            logger.info(f"Optimized PDF processing took {end_time - start_time:.2f} seconds")
            
            return full_text
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return f"Error processing PDF: {str(e)}"
    
    def _process_image_buffer(self, img_buffer, page_num, cache_dir):
        """Process an image from a memory buffer and cache the result"""
        try:
            # Open image from buffer
            image = Image.open(img_buffer)
            
            # Optimize image for OCR
            image = image.convert('L')  # Convert to grayscale
            
            # Apply some basic image enhancements
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)  # Increase contrast
            
            # Use optimized Tesseract configuration
            custom_config = r'--oem 3 --psm 6 -l eng'
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # Cache the result
            cache_file = os.path.join(cache_dir, f"page_{page_num}.txt")
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            return text
        except Exception as e:
            logger.error(f"Error processing image buffer: {e}")
            return f"Error processing image: {str(e)}"
    
    def _process_text(self, text_path: str) -> str:
        """Extract text from a text file"""
        try:
            with open(text_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding if UTF-8 fails
            try:
                with open(text_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                print(f"Error processing text file: {e}")
                return f"Error processing text file: {str(e)}"
        except Exception as e:
            print(f"Error processing text file: {e}")
            return f"Error processing text file: {str(e)}"
    
    def extract_answer(self, query: str, document_chunks: List[Dict[str, Any]]) -> Dict[str, str]:
        """Extract answer from document chunks using Gemini"""
        # Combine chunks into a single context
        context = ""
        for chunk in document_chunks:
            context += chunk["text"] + "\n\n"
        
        # Create prompt for answer extraction
        prompt = f"""
        Based on the following document content, answer the question.
        If the answer cannot be found in the document, say "No relevant information found."
        
        QUESTION: {query}
        
        DOCUMENT CONTENT:
        {context}
        
        Provide a concise answer with a citation that includes the relevant part of the document.
        Format your response as follows:
        ANSWER: [Your answer here]
        CITATION: [Relevant excerpt from the document]
        """
        
        try:
            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Parse answer and citation
            answer_match = re.search(r"ANSWER:\s*(.+?)(?=CITATION:|$)", response_text, re.DOTALL)
            citation_match = re.search(r"CITATION:\s*(.+?)$", response_text, re.DOTALL)
            
            answer = answer_match.group(1).strip() if answer_match else "No answer generated"
            citation = citation_match.group(1).strip() if citation_match else "No citation available"
            
            return {
                "answer": answer,
                "citation": citation
            }
        except Exception as e:
            print(f"Error extracting answer: {e}")
            return {
                "answer": "Error generating answer",
                "citation": f"Error: {str(e)}"
            }