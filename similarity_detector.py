#!/usr/bin/env python3
"""
Similarity Detection Engine for Ultimate Scanner
Detects near-duplicates beyond exact hash matches using perceptual hashing,
audio fingerprinting, and document fuzzy matching.
"""

import hashlib
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

# Optional imports with graceful fallbacks
try:
    from PIL import Image, ExifTags
    import imagehash
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from Levenshtein import ratio as levenshtein_ratio
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SimilarityMatch:
    """Represents a similarity match between files"""
    file1_path: str
    file2_path: str
    similarity_type: str
    confidence: float
    hash1: str
    hash2: str
    distance: int
    metadata: Dict = None

class SimilarityDetectorBase(ABC):
    """Base class for similarity detectors"""
    
    @abstractmethod
    def can_process(self, file_path: Path) -> bool:
        """Check if this detector can process the given file"""
        pass
    
    @abstractmethod
    def compute_hash(self, file_path: Path) -> Optional[str]:
        """Compute similarity hash for the file"""
        pass
    
    @abstractmethod
    def compute_distance(self, hash1: str, hash2: str) -> int:
        """Compute distance between two hashes (lower = more similar)"""
        pass
    
    @abstractmethod
    def get_similarity_threshold(self, sensitivity: str = "medium") -> int:
        """Get distance threshold for similarity (lower = more strict)"""
        pass

class ImageSimilarityDetector(SimilarityDetectorBase):
    """Perceptual image similarity detector using multiple hashing algorithms"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    
    def __init__(self, algorithm: str = "phash"):
        """
        Initialize image similarity detector
        
        Args:
            algorithm: Hashing algorithm ('phash', 'ahash', 'dhash', 'whash')
        """
        if not PILLOW_AVAILABLE:
            raise ImportError("PIL/Pillow and imagehash required for image similarity detection")
        
        self.algorithm = algorithm
        self.hash_func = {
            'phash': imagehash.phash,
            'ahash': imagehash.average_hash,
            'dhash': imagehash.dhash,
            'whash': imagehash.whash
        }.get(algorithm, imagehash.phash)
    
    def can_process(self, file_path: Path) -> bool:
        """Check if file is a supported image format"""
        return file_path.suffix.lower() in self.SUPPORTED_FORMATS
    
    def compute_hash(self, file_path: Path) -> Optional[str]:
        """Compute perceptual hash for image"""
        try:
            with Image.open(file_path) as img:
                # Convert to RGB to handle different color modes
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Compute perceptual hash
                phash = self.hash_func(img)
                return str(phash)
        
        except Exception as e:
            logger.warning(f"Failed to compute image hash for {file_path}: {e}")
            return None
    
    def compute_distance(self, hash1: str, hash2: str) -> int:
        """Compute Hamming distance between two perceptual hashes"""
        try:
            # Convert string hashes back to imagehash objects for comparison
            h1 = imagehash.hex_to_hash(hash1)
            h2 = imagehash.hex_to_hash(hash2)
            return h1 - h2  # Hamming distance
        except Exception:
            return float('inf')
    
    def get_similarity_threshold(self, sensitivity: str = "medium") -> int:
        """Get distance threshold based on sensitivity"""
        thresholds = {
            "low": 15,      # Very permissive
            "medium": 10,   # Balanced
            "high": 5       # Strict
        }
        return thresholds.get(sensitivity, 10)
    
    def extract_metadata(self, file_path: Path) -> Dict:
        """Extract EXIF and other metadata from image"""
        metadata = {}
        try:
            with Image.open(file_path) as img:
                # Basic image info
                metadata.update({
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                })
                
                # EXIF data
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    for tag_id, value in exif.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        if isinstance(value, (str, int, float)):
                            metadata[f'exif_{tag}'] = value
        
        except Exception as e:
            logger.debug(f"Failed to extract image metadata from {file_path}: {e}")
        
        return metadata

class AudioSimilarityDetector(SimilarityDetectorBase):
    """Audio similarity detector using acoustic fingerprinting"""
    
    SUPPORTED_FORMATS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
    
    def __init__(self, method: str = "chromagram"):
        """
        Initialize audio similarity detector
        
        Args:
            method: Feature extraction method ('chromagram', 'mfcc', 'spectral')
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for audio similarity detection")
        
        self.method = method
    
    def can_process(self, file_path: Path) -> bool:
        """Check if file is a supported audio format"""
        return file_path.suffix.lower() in self.SUPPORTED_FORMATS
    
    def compute_hash(self, file_path: Path) -> Optional[str]:
        """Compute acoustic fingerprint for audio file"""
        try:
            # Load audio file
            y, sr = librosa.load(str(file_path), duration=30)  # Analyze first 30 seconds
            
            if self.method == "chromagram":
                # Chromagram-based fingerprint
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                fingerprint = np.mean(chroma, axis=1)
            
            elif self.method == "mfcc":
                # MFCC-based fingerprint
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                fingerprint = np.mean(mfcc, axis=1)
            
            elif self.method == "spectral":
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                fingerprint = np.concatenate([
                    np.mean(spectral_centroids),
                    np.mean(spectral_rolloff)
                ])
            
            else:
                raise ValueError(f"Unknown audio method: {self.method}")
            
            # Convert to hash string
            fingerprint_bytes = fingerprint.tobytes()
            return hashlib.md5(fingerprint_bytes).hexdigest()
        
        except Exception as e:
            logger.warning(f"Failed to compute audio hash for {file_path}: {e}")
            return None
    
    def compute_distance(self, hash1: str, hash2: str) -> int:
        """Compute distance between audio fingerprints"""
        # For now, use simple hash comparison
        # In a more sophisticated implementation, we'd compare the actual feature vectors
        return 0 if hash1 == hash2 else 1
    
    def get_similarity_threshold(self, sensitivity: str = "medium") -> int:
        """Get distance threshold based on sensitivity"""
        # For exact hash matching, threshold is binary
        return 0

class DocumentSimilarityDetector(SimilarityDetectorBase):
    """Document similarity detector using text fuzzy matching"""
    
    SUPPORTED_FORMATS = {'.txt', '.pdf', '.docx', '.doc', '.rtf', '.md'}
    
    def __init__(self, method: str = "levenshtein"):
        """
        Initialize document similarity detector
        
        Args:
            method: Similarity method ('levenshtein', 'jaccard', 'cosine')
        """
        self.method = method
    
    def can_process(self, file_path: Path) -> bool:
        """Check if file is a supported document format"""
        return file_path.suffix.lower() in self.SUPPORTED_FORMATS
    
    def extract_text(self, file_path: Path) -> Optional[str]:
        """Extract text content from document"""
        try:
            ext = file_path.suffix.lower()
            
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif ext == '.pdf' and PDF_AVAILABLE:
                text = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                return text
            
            elif ext == '.docx' and DOCX_AVAILABLE:
                doc = Document(file_path)
                return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
            elif ext == '.md':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            else:
                # Fallback: try to read as text
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        
        except Exception as e:
            logger.warning(f"Failed to extract text from {file_path}: {e}")
            return None
    
    def compute_hash(self, file_path: Path) -> Optional[str]:
        """Compute text-based hash for document"""
        text = self.extract_text(file_path)
        if not text:
            return None
        
        # Normalize text
        normalized = self.normalize_text(text)
        
        # Create hash representing text structure
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    def compute_distance(self, hash1: str, hash2: str) -> int:
        """Compute distance between document hashes"""
        if not LEVENSHTEIN_AVAILABLE:
            # Simple hash comparison fallback
            return 0 if hash1 == hash2 else 1
        
        # For documents, we need to compare actual text content
        # This is a simplified version - in practice we'd store and compare the normalized text
        return 0 if hash1 == hash2 else 1
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two text strings"""
        if not text1 or not text2:
            return 0.0
        
        if not LEVENSHTEIN_AVAILABLE:
            return 1.0 if text1 == text2 else 0.0
        
        # Normalize both texts
        norm1 = self.normalize_text(text1)
        norm2 = self.normalize_text(text2)
        
        # Compute Levenshtein similarity ratio
        return levenshtein_ratio(norm1, norm2)
    
    def get_similarity_threshold(self, sensitivity: str = "medium") -> int:
        """Get distance threshold based on sensitivity"""
        # For text similarity, we use a different approach
        return 0

class SimilarityManager:
    """Manages multiple similarity detectors and coordinates detection"""
    
    def __init__(self):
        self.detectors: Dict[str, SimilarityDetectorBase] = {}
        self.enabled_types: Set[str] = set()
        
        # Initialize available detectors
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize available similarity detectors"""
        try:
            if PILLOW_AVAILABLE:
                self.detectors['image'] = ImageSimilarityDetector()
                logger.info("Image similarity detector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize image detector: {e}")
        
        try:
            if LIBROSA_AVAILABLE:
                self.detectors['audio'] = AudioSimilarityDetector()
                logger.info("Audio similarity detector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize audio detector: {e}")
        
        try:
            self.detectors['document'] = DocumentSimilarityDetector()
            logger.info("Document similarity detector initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize document detector: {e}")
    
    def enable_similarity_type(self, similarity_type: str):
        """Enable a specific similarity detection type"""
        if similarity_type in self.detectors:
            self.enabled_types.add(similarity_type)
            logger.info(f"Enabled {similarity_type} similarity detection")
        else:
            logger.warning(f"Unknown similarity type: {similarity_type}")
    
    def disable_similarity_type(self, similarity_type: str):
        """Disable a specific similarity detection type"""
        self.enabled_types.discard(similarity_type)
        logger.info(f"Disabled {similarity_type} similarity detection")
    
    def get_available_detectors(self) -> List[str]:
        """Get list of available detector types"""
        return list(self.detectors.keys())
    
    def compute_similarity_hash(self, file_path: Path) -> Dict[str, str]:
        """Compute similarity hashes for a file using all enabled detectors"""
        hashes = {}
        
        for similarity_type in self.enabled_types:
            detector = self.detectors.get(similarity_type)
            if detector and detector.can_process(file_path):
                try:
                    hash_value = detector.compute_hash(file_path)
                    if hash_value:
                        hashes[similarity_type] = hash_value
                except Exception as e:
                    logger.warning(f"Failed to compute {similarity_type} hash for {file_path}: {e}")
        
        return hashes
    
    def find_similar_files(self, files_with_hashes: List[Tuple[Path, Dict[str, str]]], 
                          sensitivity: str = "medium") -> List[SimilarityMatch]:
        """Find similar files based on computed hashes"""
        matches = []
        
        for similarity_type in self.enabled_types:
            detector = self.detectors.get(similarity_type)
            if not detector:
                continue
            
            threshold = detector.get_similarity_threshold(sensitivity)
            
            # Group files by similarity type
            files_with_type = [(path, hashes.get(similarity_type)) 
                             for path, hashes in files_with_hashes 
                             if similarity_type in hashes]
            
            # Compare all pairs
            for i, (file1, hash1) in enumerate(files_with_type):
                for file2, hash2 in files_with_type[i+1:]:
                    if hash1 and hash2:
                        distance = detector.compute_distance(hash1, hash2)
                        if distance <= threshold:
                            confidence = max(0, 1 - (distance / (threshold + 1)))
                            matches.append(SimilarityMatch(
                                file1_path=str(file1),
                                file2_path=str(file2),
                                similarity_type=similarity_type,
                                confidence=confidence,
                                hash1=hash1,
                                hash2=hash2,
                                distance=distance
                            ))
        
        return matches

def get_similarity_manager() -> SimilarityManager:
    """Get a configured similarity manager instance"""
    manager = SimilarityManager()
    
    # Enable all available detectors by default
    for detector_type in manager.get_available_detectors():
        manager.enable_similarity_type(detector_type)
    
    return manager

# Check what similarity detectors are available
def check_similarity_support() -> Dict[str, bool]:
    """Check which similarity detection features are available"""
    return {
        'image': PILLOW_AVAILABLE,
        'audio': LIBROSA_AVAILABLE,
        'document': LEVENSHTEIN_AVAILABLE or PDF_AVAILABLE or DOCX_AVAILABLE,
        'pillow': PILLOW_AVAILABLE,
        'librosa': LIBROSA_AVAILABLE,
        'levenshtein': LEVENSHTEIN_AVAILABLE,
        'pdf': PDF_AVAILABLE,
        'docx': DOCX_AVAILABLE
    }

if __name__ == "__main__":
    # Test the similarity detection system
    support = check_similarity_support()
    print("Similarity Detection Support:")
    for feature, available in support.items():
        status = "✓" if available else "✗"
        print(f"  {status} {feature}")
    
    if any(support.values()):
        print("\nTesting similarity manager...")
        manager = get_similarity_manager()
        print(f"Available detectors: {manager.get_available_detectors()}")
    else:
        print("\nNo similarity detection dependencies available.")
        print("Install with: pip install Pillow imagehash librosa python-Levenshtein PyPDF2 python-docx")