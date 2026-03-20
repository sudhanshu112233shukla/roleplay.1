import CoreData
import Foundation

final class CharacterRepository {
    private let ctx: NSManagedObjectContext

    init(ctx: NSManagedObjectContext = CoreDataStack.shared.viewContext) {
        self.ctx = ctx
    }

    func list() -> [CharacterProfile] {
        let req = NSFetchRequest<NSManagedObject>(entityName: "CDCharacter")
        req.sortDescriptors = [NSSortDescriptor(key: "name", ascending: true)]
        let rows = (try? ctx.fetch(req)) ?? []
        return rows.compactMap { row in
            guard
                let id = row.value(forKey: "id") as? String,
                let name = row.value(forKey: "name") as? String,
                let personality = row.value(forKey: "personality") as? String,
                let world = row.value(forKey: "world") as? String,
                let tone = row.value(forKey: "tone") as? String
            else { return nil }
            return CharacterProfile(id: id, name: name, personality: personality, world: world, tone: tone)
        }
    }

    func get(id: String) -> CharacterProfile? {
        let req = NSFetchRequest<NSManagedObject>(entityName: "CDCharacter")
        req.predicate = NSPredicate(format: "id == %@", id)
        req.fetchLimit = 1
        guard let row = (try? ctx.fetch(req))?.first else { return nil }
        guard
            let rid = row.value(forKey: "id") as? String,
            let name = row.value(forKey: "name") as? String,
            let personality = row.value(forKey: "personality") as? String,
            let world = row.value(forKey: "world") as? String,
            let tone = row.value(forKey: "tone") as? String
        else { return nil }
        return CharacterProfile(id: rid, name: name, personality: personality, world: world, tone: tone)
    }

    func upsert(_ character: CharacterProfile) {
        let req = NSFetchRequest<NSManagedObject>(entityName: "CDCharacter")
        req.predicate = NSPredicate(format: "id == %@", character.id)
        req.fetchLimit = 1

        let row: NSManagedObject
        if let existing = (try? ctx.fetch(req))?.first {
            row = existing
        } else {
            let entity = NSEntityDescription.entity(forEntityName: "CDCharacter", in: ctx)!
            row = NSManagedObject(entity: entity, insertInto: ctx)
            row.setValue(character.id, forKey: "id")
        }

        row.setValue(character.name, forKey: "name")
        row.setValue(character.personality, forKey: "personality")
        row.setValue(character.world, forKey: "world")
        row.setValue(character.tone, forKey: "tone")
        CoreDataStack.shared.saveIfNeeded()
    }

    func delete(id: String) {
        let req = NSFetchRequest<NSManagedObject>(entityName: "CDCharacter")
        req.predicate = NSPredicate(format: "id == %@", id)
        if let rows = try? ctx.fetch(req) {
            for r in rows { ctx.delete(r) }
            CoreDataStack.shared.saveIfNeeded()
        }
    }

    func ensureDefaults() {
        let existing = Set(list().map(\.id))
        if !existing.contains("wizard") {
            upsert(
                CharacterProfile(
                    id: "wizard",
                    name: "Wizard",
                    personality: "Ancient magical teacher; wise and mysterious.",
                    world: "Fantasy",
                    tone: "Wise, descriptive, emotionally consistent."
                )
            )
        }
        if !existing.contains("detective") {
            upsert(
                CharacterProfile(
                    id: "detective",
                    name: "Detective",
                    personality: "Sharp, skeptical, detail-oriented investigator.",
                    world: "Noir city",
                    tone: "Concise, probing questions, grounded."
                )
            )
        }
        if !existing.contains("historian") {
            upsert(
                CharacterProfile(
                    id: "historian",
                    name: "Historian",
                    personality: "Curious scholar who cites context and asks clarifying questions.",
                    world: "Any era you choose",
                    tone: "Clear, reflective, precise."
                )
            )
        }
        if !existing.contains("space_captain") {
            upsert(
                CharacterProfile(
                    id: "space_captain",
                    name: "Space Captain",
                    personality: "Decisive leader; pragmatic; protective of crew.",
                    world: "Deep space frontier",
                    tone: "Confident, tactical, cinematic."
                )
            )
        }
    }
}
