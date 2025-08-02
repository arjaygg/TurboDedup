#!/usr/bin/env python3
"""
Auto-Configuration Script for Ultimate Scanner
Detects your system and recommends optimal settings
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

class SystemDetector:
    """Detect system characteristics for optimal configuration"""
    
    def __init__(self):
        self.system = platform.system()
        self.cpu_count = os.cpu_count() or 4
        self.is_windows = self.system == "Windows"
        self.is_linux = self.system == "Linux"
        self.is_mac = self.system == "Darwin"
    
    def detect_storage_type(self, path: str = None) -> str:
        """Detect if path is on SSD or HDD"""
        if path is None:
            path = os.getcwd()
        
        try:
            if self.is_windows:
                # Windows detection using WMI
                drive = Path(path).drive
                if not drive:
                    drive = "C:"
                
                # Try PowerShell first (more reliable)
                try:
                    cmd = f'powershell "Get-PhysicalDisk | Where-Object {{$_.DeviceID -eq (Get-Partition -DriveLetter {drive[0]}).DiskNumber}} | Select-Object -ExpandProperty MediaType"'
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
                    if "SSD" in result.stdout:
                        return "SSD"
                    elif "HDD" in result.stdout:
                        return "HDD"
                except:
                    pass
                
                # Fallback to WMI
                try:
                    result = subprocess.run(
                        'wmic diskdrive get model,mediatype',
                        shell=True, capture_output=True, text=True, timeout=5
                    )
                    # Simple heuristic: if any drive is SSD, assume we're on SSD
                    if "SSD" in result.stdout or "Solid" in result.stdout:
                        return "SSD"
                except:
                    pass
            
            elif self.is_linux:
                # Linux detection
                try:
                    # Check if path is on an SSD
                    result = subprocess.run(
                        ['lsblk', '-d', '-o', 'name,rota'],
                        capture_output=True, text=True
                    )
                    # ROTA=0 means SSD, ROTA=1 means HDD
                    lines = result.stdout.strip().split('\n')[1:]  # Skip header
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 2 and parts[1] == '0':
                            return "SSD"
                    return "HDD"
                except:
                    pass
            
            elif self.is_mac:
                # macOS detection
                try:
                    result = subprocess.run(
                        ['diskutil', 'info', '/'],
                        capture_output=True, text=True
                    )
                    if "Solid State" in result.stdout:
                        return "SSD"
                    return "HDD"
                except:
                    pass
                
        except Exception as e:
            print(f"Warning: Could not detect storage type: {e}")
        
        # Default assumption for safety (HDD settings are safer)
        return "HDD"
    
    def detect_ram_gb(self) -> int:
        """Detect system RAM in GB"""
        try:
            if self.is_windows:
                result = subprocess.run(
                    'wmic computersystem get TotalPhysicalMemory',
                    shell=True, capture_output=True, text=True
                )
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    try:
                        bytes_ram = int(lines[1].strip())
                        return max(1, bytes_ram // (1024**3))
                    except:
                        pass
            
            elif self.is_linux:
                # Linux
                try:
                    with open('/proc/meminfo', 'r') as f:
                        for line in f:
                            if line.startswith('MemTotal:'):
                                kb = int(line.split()[1])
                                return max(1, kb // (1024**2))
                except:
                    pass
            
            elif self.is_mac:
                # macOS
                try:
                    result = subprocess.run(
                        ['sysctl', 'hw.memsize'],
                        capture_output=True, text=True
                    )
                    bytes_ram = int(result.stdout.split()[1])
                    return max(1, bytes_ram // (1024**3))
                except:
                    pass
                    
        except Exception as e:
            print(f"Warning: Could not detect RAM: {e}")
        
        return 8  # Default assumption
    
    def detect_free_space_gb(self, path: str = None) -> int:
        """Detect free space in GB"""
        if path is None:
            path = os.getcwd()
        
        try:
            if self.is_windows:
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(path), 
                    ctypes.pointer(free_bytes), 
                    None, None
                )
                return max(1, free_bytes.value // (1024**3))
            else:
                # Unix-like systems
                stat = os.statvfs(path)
                return max(1, (stat.f_bavail * stat.f_frsize) // (1024**3))
        except Exception as e:
            print(f"Warning: Could not detect free space: {e}")
            return 100  # Default assumption

    def is_network_drive(self, path: str = None) -> bool:
        """Check if path is on a network drive"""
        if path is None:
            path = os.getcwd()
        
        try:
            if self.is_windows:
                # Check for UNC path
                if path.startswith('\\\\'):
                    return True
                
                # Check if drive is network mapped
                drive = Path(path).drive
                if drive:
                    result = subprocess.run(
                        f'net use {drive}',
                        shell=True, capture_output=True, text=True
                    )
                    if "OK" in result.stdout and "\\\\" in result.stdout:
                        return True
                
                # Simple heuristic: drives after E: are often network/USB
                if drive and drive[0].upper() > 'E':
                    # Ask user to confirm
                    response = input(f"Is {drive} a network drive? (y/n): ").lower()
                    return response == 'y'
                    
            else:
                # Unix-like: check mount points
                if any(mount in path for mount in ['/mnt/', '/media/', '/Volumes/']):
                    return True
                if path.startswith('//'):
                    return True
                    
        except:
            pass
        
        return False

class ConfigOptimizer:
    """Generate optimal configuration based on system detection"""
    
    def __init__(self):
        self.detector = SystemDetector()
    
    def get_optimal_config(self, scan_path: str = None) -> dict:
        """Generate optimal configuration for the system"""
        
        # Detect system characteristics
        print("Analyzing system characteristics...")
        storage_type = self.detector.detect_storage_type(scan_path)
        ram_gb = self.detector.detect_ram_gb()
        free_space_gb = self.detector.detect_free_space_gb(scan_path)
        is_network = self.detector.is_network_drive(scan_path)
        cpu_count = self.detector.cpu_count
        
        config = {}
        
        # Determine workers based on storage type and CPU
        if is_network:
            config['workers'] = min(4, cpu_count)
            print("  → Network drive detected, limiting workers")
        elif storage_type == "SSD":
            # SSDs can handle many parallel operations
            if ram_gb >= 16:
                config['workers'] = min(cpu_count * 2, 32)
            else:
                config['workers'] = min(cpu_count * 2, 16)
        else:  # HDD
            # HDDs perform poorly with too many parallel operations
            config['workers'] = min(4, cpu_count)
        
        # Determine chunk size based on RAM and storage
        if is_network:
            config['chunk_size'] = "256KB"  # Smaller chunks for network
        elif storage_type == "SSD":
            if ram_gb >= 16:
                config['chunk_size'] = "4MB"
            elif ram_gb >= 8:
                config['chunk_size'] = "2MB"
            else:
                config['chunk_size'] = "1MB"
        else:  # HDD
            config['chunk_size'] = "1MB"  # Balanced for HDD
        
        # Determine min size based on free space and use case
        if free_space_gb < 10:
            config['min_size'] = "1GB"  # Only find very large duplicates
            print("  → Low disk space, focusing on large files only")
        elif free_space_gb < 50:
            config['min_size'] = "500MB"
        elif free_space_gb < 200:
            config['min_size'] = "100MB"
        else:
            config['min_size'] = "50MB"
        
        # Determine partial hash threshold
        if is_network:
            config['partial_threshold'] = "1GB"  # Minimize network reads
        elif storage_type == "SSD":
            config['partial_threshold'] = "256MB"  # SSDs are fast
        else:  # HDD
            config['partial_threshold'] = "512MB"
        
        # Determine segment size for partial hash
        if is_network:
            config['partial_segment'] = "4MB"  # Smaller for network
        else:
            config['partial_segment'] = "8MB"  # Default
        
        # Determine algorithm
        try:
            import xxhash
            has_xxhash = True
        except ImportError:
            has_xxhash = False
        
        if has_xxhash and ram_gb >= 8 and cpu_count >= 4:
            config['algorithm'] = "xxhash"
        else:
            config['algorithm'] = "md5"
        
        # Additional optimizations
        if is_network:
            config['retry_attempts'] = 5
            config['batch_size'] = 500
        else:
            config['retry_attempts'] = 2
            config['batch_size'] = 1000
        
        return config
    
    def generate_command(self, config: dict, scan_path: str = None) -> str:
        """Generate the optimal command line"""
        cmd_parts = ["python", "ultimate_scanner.py"]
        
        if scan_path:
            # Handle spaces in path
            if ' ' in scan_path:
                cmd_parts.extend(["--path", f'"{scan_path}"'])
            else:
                cmd_parts.extend(["--path", scan_path])
        
        # Add configuration parameters
        param_map = {
            'workers': '--workers',
            'chunk_size': '--chunk-size',
            'min_size': '--min-size',
            'partial_threshold': '--partial-threshold',
            'partial_segment': '--partial-segment',
            'algorithm': '--algorithm',
            'retry_attempts': '--retry-attempts',
            'batch_size': '--batch-size',
        }
        
        for key, value in config.items():
            if key in param_map:
                cmd_parts.extend([param_map[key], str(value)])
        
        return " ".join(cmd_parts)
    
    def explain_config(self, config: dict, scan_path: str = None) -> None:
        """Explain why each setting was chosen"""
        storage = self.detector.detect_storage_type(scan_path)
        ram = self.detector.detect_ram_gb()
        free = self.detector.detect_free_space_gb(scan_path)
        network = self.detector.is_network_drive(scan_path)
        
        print("\n" + "="*60)
        print("SYSTEM DETECTION RESULTS")
        print("="*60)
        print(f"Operating System: {self.detector.system}")
        print(f"CPU Cores: {self.detector.cpu_count}")
        print(f"System RAM: {ram} GB")
        print(f"Storage Type: {storage}")
        print(f"Free Space: {free} GB")
        print(f"Network Drive: {'Yes' if network else 'No'}")
        
        print("\n" + "="*60)
        print("OPTIMAL CONFIGURATION EXPLANATION")
        print("="*60)
        
        # Workers explanation
        print(f"\n1. Workers: {config['workers']}")
        if network:
            print("   → Limited to 4 due to network latency")
            print("   → More workers would cause congestion")
        elif storage == "SSD":
            print("   → SSD can handle parallel I/O efficiently")
            print(f"   → Using {config['workers']} workers (CPU×2, capped)")
        else:
            print("   → HDD limited to 4 to prevent disk thrashing")
            print("   → Sequential access is faster on spinning disks")
        
        # Chunk size explanation
        print(f"\n2. Chunk Size: {config['chunk_size']}")
        if network:
            print("   → Smaller chunks reduce network round-trips")
        elif storage == "SSD":
            print("   → Larger chunks for SSD sequential speed")
            if ram >= 16:
                print("   → High RAM allows larger buffers")
        else:
            print("   → 1MB is optimal for HDD read patterns")
        
        # Min size explanation
        print(f"\n3. Minimum File Size: {config['min_size']}")
        if free < 50:
            print(f"   → Limited free space ({free}GB)")
            print("   → Focusing on large files for quick wins")
        else:
            print("   → Sufficient space for thorough scanning")
            print("   → Will catch most duplicates at this threshold")
        
        # Partial hash explanation
        print(f"\n4. Partial Hash Threshold: {config['partial_threshold']}")
        if network:
            print("   → Higher threshold minimizes network reads")
            print("   → Only very large files use partial hash")
        elif storage == "SSD":
            print("   → Lower threshold - SSDs handle it well")
            print("   → More files benefit from optimization")
        else:
            print("   → Balanced for HDD performance")
        
        # Algorithm explanation
        print(f"\n5. Hash Algorithm: {config['algorithm']}")
        if config['algorithm'] == "xxhash":
            print("   → xxHash is 2-3x faster than MD5")
            print("   → Good CPU and RAM support it")
        else:
            print("   → MD5 is fast and widely compatible")
            if 'xxhash' not in config['algorithm']:
                print("   → (Install xxhash for even better speed)")

def main():
    """Main function to run auto-configuration"""
    print("Ultimate Scanner Auto-Configuration Tool")
    print("========================================\n")
    
    # Get scan path
    if len(sys.argv) > 1:
        scan_path = sys.argv[1]
    else:
        scan_path = input("Enter path to scan (or press Enter for current directory): ").strip()
        if not scan_path:
            scan_path = os.getcwd()
    
    # Validate path
    if not os.path.exists(scan_path):
        print(f"\nError: Path '{scan_path}' does not exist!")
        sys.exit(1)
    
    scan_path = os.path.abspath(scan_path)
    print(f"\nAnalyzing optimal settings for: {scan_path}")
    
    # Generate configuration
    optimizer = ConfigOptimizer()
    config = optimizer.get_optimal_config(scan_path)
    
    # Explain the configuration
    optimizer.explain_config(config, scan_path)
    
    # Generate command
    command = optimizer.generate_command(config, scan_path)
    
    print("\n" + "="*60)
    print("RECOMMENDED COMMAND")
    print("="*60)
    print(f"\n{command}\n")
    
    print("="*60)
    print("QUICK ALTERNATIVES")
    print("="*60)
    print("\nFor a faster scan (large files only):")
    fast_config = config.copy()
    fast_config['min_size'] = "1GB"
    fast_config['algorithm'] = "xxhash"
    fast_cmd = optimizer.generate_command(fast_config, scan_path)
    print(f"  {fast_cmd}")
    
    print("\nFor a thorough scan (find more duplicates):")
    thorough_config = config.copy()
    thorough_config['min_size'] = "10MB"
    thorough_cmd = optimizer.generate_command(thorough_config, scan_path)
    print(f"  {thorough_cmd}")
    
    print("\n" + "="*60)
    
    # Ask if user wants to run it
    response = input("\nRun the recommended command now? (y/n): ").strip().lower()
    if response == 'y':
        print("\nStarting scan with optimal configuration...")
        print("Press Ctrl+C to stop at any time.\n")
        os.system(command)
    else:
        # Save configuration
        save = input("\nSave configuration to file? (y/n): ").strip().lower()
        if save == 'y':
            config_filename = "scan_config.txt"
            with open(config_filename, "w") as f:
                f.write(f"# Ultimate Scanner Configuration\n")
                f.write(f"# Generated for: {scan_path}\n")
                f.write(f"# Date: {platform.node()} - {os.getcwd()}\n\n")
                f.write(f"# System detected:\n")
                f.write(f"# - Storage: {optimizer.detector.detect_storage_type(scan_path)}\n")
                f.write(f"# - RAM: {optimizer.detector.detect_ram_gb()} GB\n")
                f.write(f"# - CPU cores: {optimizer.detector.cpu_count}\n\n")
                f.write("# Recommended command:\n")
                f.write(command + "\n\n")
                f.write("# Fast scan alternative:\n")
                f.write(fast_cmd + "\n\n")
                f.write("# Thorough scan alternative:\n")
                f.write(thorough_cmd + "\n")
            
            print(f"\nConfiguration saved to: {config_filename}")
            print("You can run the command later by copying from this file.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)