import SwiftUI

@main
struct DarkAIApp: App {
    init() {
        setupDirectories()
    }
    
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
    
    private func setupDirectories() {
        guard let docsUrl = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let modelsDir = docsUrl.appendingPathComponent("Models")
        
        do {
            try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
            print("Successfully initialized local models directory at: \(modelsDir.path)")
        } catch {
            print("Failed to initialize models directory: \(error.localizedDescription)")
        }
    }
}
