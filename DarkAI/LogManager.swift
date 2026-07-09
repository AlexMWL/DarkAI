import Foundation
import Combine

class LogManager: ObservableObject {
    static let shared = LogManager()
    
    @Published var logs: [String] = []
    
    private let logFileURL: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("diagnostic_logs.txt")
    }()
    
    private let formatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm:ss"
        return f
    }()
    
    private let queue = DispatchQueue(label: "com.darkai.logmanager", qos: .background)
    
    private init() {
        loadLogs()
        log("Diagnostic Logger initialized.")
    }
    
    func log(_ message: String) {
        let timestamp = formatter.string(from: Date())
        let logLine = "[\(timestamp)] \(message)"
        
        // Print to standard console so it's visible in Xcode debugging too
        print(logLine)
        
        queue.async {
            self.appendToFile(logLine)
            
            DispatchQueue.main.async {
                self.logs.append(logLine)
                if self.logs.count > 1000 {
                    self.logs.removeFirst(self.logs.count - 1000)
                }
            }
        }
    }
    
    private func loadLogs() {
        if FileManager.default.fileExists(atPath: logFileURL.path) {
            if let content = try? String(contentsOf: logFileURL, encoding: .utf8) {
                let lines = content.components(separatedBy: "\n").filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
                DispatchQueue.main.async {
                    self.logs = Array(lines.suffix(1000))
                }
            }
        }
    }
    

    
    private func appendToFile(_ line: String) {
        let data = (line + "\n").data(using: .utf8) ?? Data()
        if FileManager.default.fileExists(atPath: logFileURL.path) {
            if let fileHandle = try? FileHandle(forWritingTo: logFileURL) {
                fileHandle.seekToEndOfFile()
                fileHandle.write(data)
                fileHandle.closeFile()
            }
        } else {
            try? (line + "\n").write(to: logFileURL, atomically: true, encoding: .utf8)
        }
    }
    
    func clearLogs() {
        queue.async {
            try? "".write(to: self.logFileURL, atomically: true, encoding: .utf8)
            DispatchQueue.main.async {
                self.logs.removeAll()
                self.log("Logs cleared by user.")
            }
        }
    }
    
    func getLogFileURL() -> URL {
        return logFileURL
    }
}
