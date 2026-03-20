import Foundation

final class AppState: ObservableObject {
    private static let activeCharacterKey = "activeCharacterId"

    @Published var activeSessionId: UUID? = nil
    @Published var activeCharacterId: String {
        didSet { UserDefaults.standard.set(activeCharacterId, forKey: Self.activeCharacterKey) }
    }

    init() {
        self.activeCharacterId = UserDefaults.standard.string(forKey: Self.activeCharacterKey) ?? "wizard"
    }
}
