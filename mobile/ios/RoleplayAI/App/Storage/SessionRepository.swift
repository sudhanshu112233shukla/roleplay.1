import CoreData
import Foundation

final class SessionRepository {
    private let ctx: NSManagedObjectContext

    init(ctx: NSManagedObjectContext = CoreDataStack.shared.viewContext) {
        self.ctx = ctx
    }

    func listSessions() -> [ChatSession] {
        let req = NSFetchRequest<NSManagedObject>(entityName: "CDSession")
        req.sortDescriptors = [NSSortDescriptor(key: "updatedAt", ascending: false)]
        let rows = (try? ctx.fetch(req)) ?? []
        return rows.compactMap { row in
            guard
                let id = row.value(forKey: "id") as? UUID,
                let title = row.value(forKey: "title") as? String,
                let characterId = row.value(forKey: "characterId") as? String,
                let createdAt = row.value(forKey: "createdAt") as? Date,
                let updatedAt = row.value(forKey: "updatedAt") as? Date
            else { return nil }
            return ChatSession(id: id, title: title, characterId: characterId, createdAt: createdAt, updatedAt: updatedAt)
        }
    }

    func createSession(characterId: String, title: String = "New chat") -> UUID {
        let now = Date()
        let id = UUID()
        let entity = NSEntityDescription.entity(forEntityName: "CDSession", in: ctx)!
        let row = NSManagedObject(entity: entity, insertInto: ctx)
        row.setValue(id, forKey: "id")
        row.setValue(title, forKey: "title")
        row.setValue(characterId, forKey: "characterId")
        row.setValue(now, forKey: "createdAt")
        row.setValue(now, forKey: "updatedAt")
        CoreDataStack.shared.saveIfNeeded()
        return id
    }

    func renameSession(id: UUID, title: String) {
        let req = NSFetchRequest<NSManagedObject>(entityName: "CDSession")
        req.predicate = NSPredicate(format: "id == %@", id as CVarArg)
        req.fetchLimit = 1
        if let row = (try? ctx.fetch(req))?.first {
            row.setValue(title, forKey: "title")
            row.setValue(Date(), forKey: "updatedAt")
            CoreDataStack.shared.saveIfNeeded()
        }
    }

    func deleteSession(id: UUID) {
        deleteMessages(sessionId: id)
        let req = NSFetchRequest<NSManagedObject>(entityName: "CDSession")
        req.predicate = NSPredicate(format: "id == %@", id as CVarArg)
        if let rows = try? ctx.fetch(req) {
            for r in rows { ctx.delete(r) }
            CoreDataStack.shared.saveIfNeeded()
        }
    }

    func listMessages(sessionId: UUID) -> [ChatMessage] {
        let req = NSFetchRequest<NSManagedObject>(entityName: "CDMessage")
        req.predicate = NSPredicate(format: "sessionId == %@", sessionId as CVarArg)
        req.sortDescriptors = [NSSortDescriptor(key: "createdAt", ascending: true)]
        let rows = (try? ctx.fetch(req)) ?? []
        return rows.compactMap { row in
            guard
                let id = row.value(forKey: "id") as? UUID,
                let role = row.value(forKey: "role") as? String,
                let content = row.value(forKey: "content") as? String,
                let createdAt = row.value(forKey: "createdAt") as? Date
            else { return nil }
            return ChatMessage(id: id, sessionId: sessionId, role: Role(rawValue: role) ?? .assistant, content: content, createdAt: createdAt)
        }
    }

    func addMessage(sessionId: UUID, role: Role, content: String) {
        let entity = NSEntityDescription.entity(forEntityName: "CDMessage", in: ctx)!
        let row = NSManagedObject(entity: entity, insertInto: ctx)
        row.setValue(UUID(), forKey: "id")
        row.setValue(sessionId, forKey: "sessionId")
        row.setValue(role.rawValue, forKey: "role")
        row.setValue(content, forKey: "content")
        row.setValue(Date(), forKey: "createdAt")
        touchSession(sessionId: sessionId)
        CoreDataStack.shared.saveIfNeeded()
    }

    private func touchSession(sessionId: UUID) {
        let req = NSFetchRequest<NSManagedObject>(entityName: "CDSession")
        req.predicate = NSPredicate(format: "id == %@", sessionId as CVarArg)
        req.fetchLimit = 1
        if let row = (try? ctx.fetch(req))?.first {
            row.setValue(Date(), forKey: "updatedAt")
        }
    }

    private func deleteMessages(sessionId: UUID) {
        let req = NSFetchRequest<NSManagedObject>(entityName: "CDMessage")
        req.predicate = NSPredicate(format: "sessionId == %@", sessionId as CVarArg)
        if let rows = try? ctx.fetch(req) {
            for r in rows { ctx.delete(r) }
        }
    }
}

