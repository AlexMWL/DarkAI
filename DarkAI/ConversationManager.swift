//
//  ConversationManager.swift
//  DarkAI
//
//  Created by Antigravity on 6/28/26.
//

import Foundation
import Combine

struct ChatMessage: Identifiable, Codable, Equatable {
    var id: UUID = UUID()
    var isUser: Bool
    var text: String
    var timestamp: Date = Date()
}

struct Conversation: Identifiable, Codable, Equatable {
    var id: UUID = UUID()
    var title: String
    var messages: [ChatMessage]
    var createdAt: Date = Date()
}

class ConversationManager: ObservableObject {
    @Published var conversations: [Conversation] = []
    @Published var activeConversationId: UUID? = nil
    @Published var stealthMode: Bool = false
    
    private let storageKey = "DarkAI_Conversations"
    
    init() {
        loadConversations()
    }
    
    func loadConversations() {
        if let data = UserDefaults.standard.data(forKey: storageKey),
           let decoded = try? JSONDecoder().decode([Conversation].self, from: data) {
            self.conversations = decoded
            if let first = decoded.first {
                self.activeConversationId = first.id
            }
        } else {
            // Seed a default chat
            let firstChat = Conversation(
                title: "DarkAI Launch Log",
                messages: [
                    ChatMessage(isUser: false, text: "Welcome to DarkAI. System bypass active. Load a GGUF model in settings to begin local on-device generation.")
                ]
            )
            self.conversations = [firstChat]
            self.activeConversationId = firstChat.id
            saveConversations()
        }
    }
    
    func saveConversations() {
        guard !stealthMode else { return } // Do not save in stealth mode
        if let encoded = try? JSONEncoder().encode(conversations) {
            UserDefaults.standard.set(encoded, forKey: storageKey)
        }
    }
    
    var activeConversation: Conversation? {
        conversations.first(where: { $0.id == activeConversationId })
    }
    
    func createConversation() {
        let newChat = Conversation(title: "Chat \(conversations.count + 1)", messages: [])
        conversations.insert(newChat, at: 0)
        activeConversationId = newChat.id
        saveConversations()
    }
    
    func deleteConversation(id: UUID) {
        conversations.removeAll(where: { $0.id == id })
        if activeConversationId == id {
            activeConversationId = conversations.first?.id
        }
        if conversations.isEmpty {
            createConversation()
        } else {
            saveConversations()
        }
    }
    
    func selectConversation(id: UUID) {
        activeConversationId = id
    }
    
    func addMessageToActive(isUser: Bool, text: String) {
        guard let activeId = activeConversationId else { return }
        
        if let index = conversations.firstIndex(where: { $0.id == activeId }) {
            let msg = ChatMessage(isUser: isUser, text: text)
            conversations[index].messages.append(msg)
            
            // Auto update title if it was default and has messages now
            if conversations[index].title.hasPrefix("Chat ") && isUser {
                let cleanTitle = text.prefix(20).trimmingCharacters(in: .whitespacesAndNewlines)
                conversations[index].title = cleanTitle.isEmpty ? "Chat" : String(cleanTitle) + "..."
            }
            
            saveConversations()
            
            // Trigger manual object update notification
            objectWillChange.send()
        }
    }
    
    func updateLastMessage(text: String) {
        guard let activeId = activeConversationId else { return }
        if let index = conversations.firstIndex(where: { $0.id == activeId }),
           !conversations[index].messages.isEmpty {
            let lastIndex = conversations[index].messages.count - 1
            conversations[index].messages[lastIndex].text = text
            saveConversations()
            objectWillChange.send()
        }
    }
}
