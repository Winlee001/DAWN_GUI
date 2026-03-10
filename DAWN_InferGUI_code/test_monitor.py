"""
Simple test script for System Monitor functionality
"""
import sys
import time

# Import only the SystemMonitor class, not the thread class
sys.path.insert(0, '.')
from GUI_Utils.SystemMonitor import SystemMonitor

def test_system_monitor():
    """Test the SystemMonitor class"""
    print("=" * 60)
    print("Testing System Monitor")
    print("=" * 60)
    
    monitor = SystemMonitor()
    
    print(f"\nGPU Available: {monitor.gpu_available}")
    if monitor.gpu_available:
        print(f"GPU Count: {monitor.gpu_count}")
    
    print("\nCollecting metrics for 5 seconds...")
    for i in range(5):
        print(f"\n--- Iteration {i+1} ---")
        metrics = monitor.get_all_metrics()
        
        # Print CPU metrics
        print(f"CPU Usage: {metrics.cpu_percent:.1f}%")
        print(f"CPU Memory: {metrics.cpu_memory_percent:.1f}% ({metrics.cpu_memory_used_gb:.2f}/{metrics.cpu_memory_total_gb:.2f} GB)")
        
        # Print GPU metrics
        if metrics.gpu_count > 0:
            for gpu in metrics.gpu_metrics:
                print(f"\nGPU {gpu['device_id']}: {gpu['device_name']}")
                print(f"  GPU Utilization: {gpu['gpu_util']}%")
                print(f"  Memory: {gpu['memory_percent']:.1f}% ({gpu['memory_used_gb']:.2f}/{gpu['memory_total_gb']:.2f} GB)")
                if gpu['temperature'] > 0:
                    print(f"  Temperature: {gpu['temperature']}°C")
                if gpu['power_usage'] > 0:
                    print(f"  Power Usage: {gpu['power_usage']:.1f}W")
        else:
            print("No GPU metrics available")
        
        time.sleep(1)
    
    monitor.shutdown()
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_system_monitor()
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
