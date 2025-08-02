#!/usr/bin/env python3
"""
Test script for smart symlink pattern detection
"""

import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Set, Optional
import sys
import os

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

@dataclass
class MockDuplicateGroup:
    """Mock DuplicateGroup for testing"""
    size: int
    hash_val: str
    paths: List[Path]
    selected_for_deletion: Set[int] = field(default_factory=set)
    selected_for_symlink: Set[int] = field(default_factory=set)
    symlink_target_index: Optional[int] = None
    
    @property
    def count(self) -> int:
        return len(self.paths)

def test_symlink_safety_analyzer():
    """Test the SymlinkSafetyAnalyzer with various file patterns"""
    try:
        from turbodedup.cli.main import SymlinkSafetyAnalyzer
        
        analyzer = SymlinkSafetyAnalyzer()
        
        # Test cases with expected outcomes
        test_cases = [
            {
                'name': 'Valuable media files',
                'paths': [
                    '/Users/john/Pictures/vacation.jpg',
                    '/Users/john/Pictures/vacation - Copy.jpg',
                    '/Users/john/Desktop/vacation.jpg'
                ],
                'size': 5 * 1024 * 1024,  # 5MB
                'expected_strategy': 'symlink',
                'reason': 'Valuable file type and user content directory'
            },
            {
                'name': 'Temporary cache files',
                'paths': [
                    '/tmp/cache_file.tmp',
                    '/var/cache/app_cache.cache',
                    '/build/temp_output.tmp'
                ],
                'size': 100 * 1024,  # 100KB
                'expected_strategy': 'deletion',
                'reason': 'Temporary files in cache directories'
            },
            {
                'name': 'Mixed valuable documents',
                'paths': [
                    '/Users/john/Documents/report.pdf',
                    '/Users/john/Downloads/report.pdf',
                    '/Users/john/Work/report_final.pdf'
                ],
                'size': 2 * 1024 * 1024,  # 2MB
                'expected_strategy': 'symlink',
                'reason': 'Document type in user directories'
            },
            {
                'name': 'Code files with versions',
                'paths': [
                    '/Projects/myapp/config.json',
                    '/Projects/myapp/config_v2.json',
                    '/Projects/myapp/backup/config.json'
                ],
                'size': 1024,  # 1KB
                'expected_strategy': 'symlink',
                'reason': 'Code files may have subtle differences'
            },
            {
                'name': 'Large media files',
                'paths': [
                    '/Movies/movie.mp4',
                    '/Downloads/movie.mp4'
                ],
                'size': 2 * 1024 * 1024 * 1024,  # 2GB
                'expected_strategy': 'symlink',
                'reason': 'Large valuable media files'
            }
        ]
        
        print("Testing SymlinkSafetyAnalyzer with various file patterns:")
        print("=" * 70)
        
        passed_tests = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['name']}")
            print(f"Files: {test_case['paths']}")
            
            # Create mock group
            group = MockDuplicateGroup(
                size=test_case['size'],
                hash_val="test_hash",
                paths=[Path(p) for p in test_case['paths']]
            )
            
            # Analyze group
            analysis = analyzer.analyze_group_safety(group)
            
            # Check results
            actual_strategy = analysis['recommended_strategy']
            expected_strategy = test_case['expected_strategy']
            confidence = analysis['confidence']
            reasoning = analysis['reasoning']
            
            print(f"Expected: {expected_strategy}")
            print(f"Actual: {actual_strategy} (confidence: {confidence:.1f}%)")
            print(f"Reasoning: {reasoning[0] if reasoning else 'No reasoning provided'}")
            
            if actual_strategy == expected_strategy:
                print("✓ PASS")
                passed_tests += 1
            else:
                print("✗ FAIL")
                print(f"  Expected {expected_strategy} but got {actual_strategy}")
            
            # Show detailed scores
            print(f"Scores - Symlink: {analysis['symlink_safety_score']:.1f}, Deletion: {analysis['deletion_safety_score']:.1f}")
        
        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed_tests}/{len(test_cases)} tests passed")
        
        if passed_tests == len(test_cases):
            print("🎉 All symlink pattern detection tests passed!")
            return True
        else:
            print("❌ Some tests failed. Check the pattern detection logic.")
            return False
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error testing symlink safety analyzer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_example():
    """Show example of how the smart system would work"""
    print("\n" + "=" * 70)
    print("INTEGRATION EXAMPLE")
    print("=" * 70)
    
    examples = [
        "📸 Photo duplicates → SYMLINK (preserve originals)",
        "🎬 Video files → SYMLINK (large, valuable content)", 
        "📄 Documents → SYMLINK (may contain unique content)",
        "💾 Cache files → DELETE (safe to remove)",
        "🗂️ Temp files → DELETE (designed to be temporary)",
        "⚙️ Config files → SYMLINK (subtle version differences)",
        "🔧 Code files → SYMLINK (may have important differences)"
    ]
    
    print("\nSmart recommendations would work like this:")
    for example in examples:
        print(f"  {example}")
    
    print(f"\nCommand line usage:")
    print(f"  # Enable smart symlink recommendations")
    print(f"  turbodedup --enable-symlinks --delete-strategy keep_smart")
    print(f"  ")
    print(f"  # Interactive mode shows both file quality and symlink safety")
    print(f"  turbodedup --enable-symlinks --delete-strategy interactive")
    print(f"  ")
    print(f"  # Force symlinks for all strategies when safe")
    print(f"  turbodedup --enable-symlinks --prefer-symlinks")

if __name__ == "__main__":
    print("Testing Smart Symlink Pattern Detection")
    print("=" * 50)
    
    success = test_symlink_safety_analyzer()
    test_integration_example()
    
    if success:
        print(f"\n✅ Smart symlink pattern detection is working correctly!")
        print(f"The system can now intelligently recommend symlinks vs deletion")
        print(f"based on file type, location, size, and naming patterns.")
    else:
        print(f"\n❌ Smart pattern detection needs debugging.")