
import sys
import os
import subprocess
import importlib.util

def check_python_version():
    
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 9):
        print(" Python 3.9+ required")
        return False
    print(" Python version OK")
    return True

def check_dependencies():
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
            print(f" {package} not found")
        else:
            print(f" {package} available")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_pathway():
    
    try:
        import pathway as pw
        print(f" Pathway version: {pw.__version__}")
        return True
    except ImportError:
        print(" Pathway not found")
        print("Install with: pip install pathway")
        return False

def check_docker():
    
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f" Docker: {result.stdout.strip()}")
            return True
        else:
            print(" Docker command failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(" Docker not found")
        print("Install Docker Desktop from: https://www.docker.com/products/docker-desktop/")
        return False

def check_directories():
    
    dirs = ['data', 'output', 'models', 'src']
    all_exist = True
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            print(f" Directory {dir_name} exists")
        else:
            print(f" Directory {dir_name} missing")
            all_exist = False
    
    return all_exist

def main():
    
    print("=== Setup Verification ===\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Directories", check_directories),
        ("Python Dependencies", check_dependencies),
        ("Pathway Library", check_pathway),
        ("Docker Installation", check_docker),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n--- {name} ---")
        result = check_func()
        results.append((name, result))
    
    print("\n=== Summary ===")
    all_passed = True
    for name, passed in results:
        status = " PASS" if passed else " FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n All checks passed! Ready to build Docker container.")
        print("Run: ./build.sh")
    else:
        print("\n  Some checks failed. Please fix the issues above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)