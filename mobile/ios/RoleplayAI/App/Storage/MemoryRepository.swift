import CoreData
import Foundation

final class MemoryRepository {
    private let ctx: NSManagedObjectContext

    init(ctx: NSManagedObjectContext = CoreDataStack.shared.viewContext) {
        self.ctx = ctx
    }

    func list(sessionId: UUID) -> [MemoryItem] {
        let req = NSFetchRequest<NSManagedObject>(entityName: "CDMemory")
        req.predicate = NSPredicate(format: "sessionId == %@", sessionId as CVarArg)
        req.sortDescriptors = [NSSortDescriptor(key: "createdAt", ascending: false)]
        let rows = (try? ctx.fetch(req)) ?? []
        return rows.compactMap { row in
            guard
                let id = row.value(forKey: "id") as? UUID,
                let text = row.value(forKey: "text") as? String,
                let score = row.value(forKey: "score") as? Double,
                let createdAt = row.value(forKey: "createdAt") as? Date
            else { return nil }
            return MemoryItem(id: id, sessionId: sessionId, text: text, score: score, createdAt: createdAt)
        }
    }

    func add(sessionId: UUID, text: String, score: Double) {
        let entity = NSEntityDescription.entity(forEntityName: "CDMemory", in: ctx)!
        let row = NSManagedObject(entity: entity, insertInto: ctx)
        row.setValue(UUID(), forKey: "id")
        row.setValue(sessionId, forKey: "sessionId")
        row.setValue(text, forKey: "text")
        row.setValue(score, forKey: "score")
        row.setValue(Date(), forKey: "createdAt")
        CoreDataStack.shared.saveIfNeeded()
    }

    func delete(id: UUID) {
        let req = NSFetchRequest<NSManagedObject>(entityName: "CDMemory")
        req.predicate = NSPredicate(format: "id == %@", id as CVarArg)
        if let rows = try? ctx.fetch(req) {
            for r in rows { ctx.delete(r) }
            CoreDataStack.shared.saveIfNeeded()
        }
    }
}

