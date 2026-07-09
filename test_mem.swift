import Foundation
let mem = ProcessInfo.processInfo.physicalMemory
print(Double(mem) / 1024 / 1024 / 1024)
