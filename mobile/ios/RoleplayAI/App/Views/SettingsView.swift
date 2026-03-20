import SwiftUI

struct SettingsView: View {
    @State private var temperature: Double = 0.8
    @State private var maxTokens: Double = 180
    @State private var memoryEnabled: Bool = true
    @State private var voiceEnabled: Bool = true

    var body: some View {
        Form {
            Section("Local AI") {
                Text("Offline: ON")
                Toggle("Memory", isOn: $memoryEnabled)
                Toggle("Voice", isOn: $voiceEnabled)
            }
            Section("Generation") {
                VStack(alignment: .leading) {
                    Text("Temperature: \(temperature, specifier: "%.2f")")
                    Slider(value: $temperature, in: 0...1.5)
                }
                VStack(alignment: .leading) {
                    Text("Max tokens: \(Int(maxTokens))")
                    Slider(value: $maxTokens, in: 32...512, step: 1)
                }
            }
            Section("Model") {
                Text("Model selection is wired in AI engine factory next.")
            }
        }
        .navigationTitle("Settings")
    }
}

