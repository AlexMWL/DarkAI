//
//  GlitchBackgroundView.swift
//  DarkAI
//
//  Created by Antigravity on 6/29/26.
//

import SwiftUI
import Combine

struct GlitchBackgroundView: View {
    @State private var glitchOffset1: CGFloat = 0
    @State private var glitchOffset2: CGFloat = 0
    @State private var glitchOpacity: Double = 0.0
    @State private var glitchColor: Color = .red
    
    let timer = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()
    
    var body: some View {
        ZStack {
            // Base background
            Color.black.ignoresSafeArea()
            
            // Abstract grid or tech lines
            VStack(spacing: 40) {
                ForEach(0..<20) { i in
                    Rectangle()
                        .fill(Theme.textMuted.opacity(0.05))
                        .frame(height: 1)
                }
            }
            
            // Glitching element 1 (Red shift)
            Image("circuit_brain")
                .resizable()
                .scaledToFit()
                .frame(width: 200, height: 200)
                .foregroundColor(glitchColor.opacity(glitchOpacity * 0.7))
                .offset(x: glitchOffset1, y: -glitchOffset2)
                .blendMode(.screen)
            
            // Glitching element 2 (Blue shift)
            Image("circuit_brain")
                .resizable()
                .scaledToFit()
                .frame(width: 200, height: 200)
                .foregroundColor(Color.cyan.opacity(glitchOpacity * 0.7))
                .offset(x: -glitchOffset1, y: glitchOffset2)
                .blendMode(.screen)
            
            // Core element
            Image("circuit_brain")
                .resizable()
                .scaledToFit()
                .frame(width: 200, height: 200)
                .foregroundColor(Theme.border.opacity(0.1))
            
            // Random horizontal glitch bars
            GeometryReader { geo in
                ForEach(0..<5) { _ in
                    Rectangle()
                        .fill(Color.white.opacity(glitchOpacity * 0.3))
                        .frame(width: geo.size.width, height: CGFloat.random(in: 2...10))
                        .position(x: geo.size.width / 2, y: CGFloat.random(in: 0...geo.size.height))
                        .offset(x: glitchOffset1 * 2)
                }
            }
        }
        .onReceive(timer) { _ in
            // Randomly trigger glitch effect
            if Double.random(in: 0...1) > 0.85 {
                withAnimation(.linear(duration: 0.05)) {
                    glitchOffset1 = CGFloat.random(in: -15...15)
                    glitchOffset2 = CGFloat.random(in: -5...5)
                    glitchOpacity = Double.random(in: 0.3...0.8)
                    glitchColor = Double.random(in: 0...1) > 0.5 ? .red : .purple
                }
                
                // Instantly snap back
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                    withAnimation(.linear(duration: 0.05)) {
                        glitchOffset1 = 0
                        glitchOffset2 = 0
                        glitchOpacity = 0.0
                    }
                }
            }
        }
    }
}
