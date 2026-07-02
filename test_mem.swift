import Foundation
import os

let mem = os_proc_available_memory()
print("Available memory: \(mem / (1024*1024)) MB")
