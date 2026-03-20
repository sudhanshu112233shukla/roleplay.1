import CoreData

final class CoreDataStack {
    static let shared = CoreDataStack()

    let container: NSPersistentContainer

    private init() {
        let model = CoreDataStack.makeModel()
        container = NSPersistentContainer(name: "RoleplayAI", managedObjectModel: model)
        container.loadPersistentStores { _, error in
            if let error = error {
                fatalError("CoreData load failed: \(error)")
            }
        }
        container.viewContext.mergePolicy = NSMergeByPropertyObjectTrumpMergePolicy
    }

    var viewContext: NSManagedObjectContext { container.viewContext }

    func saveIfNeeded() {
        let ctx = container.viewContext
        if ctx.hasChanges {
            try? ctx.save()
        }
    }

    private static func makeModel() -> NSManagedObjectModel {
        let model = NSManagedObjectModel()

        // Session
        let session = NSEntityDescription()
        session.name = "CDSession"
        session.managedObjectClassName = "NSManagedObject"

        let sId = NSAttributeDescription()
        sId.name = "id"
        sId.attributeType = .UUIDAttributeType
        sId.isOptional = false

        let sTitle = NSAttributeDescription()
        sTitle.name = "title"
        sTitle.attributeType = .stringAttributeType
        sTitle.isOptional = false

        let sCharacterId = NSAttributeDescription()
        sCharacterId.name = "characterId"
        sCharacterId.attributeType = .stringAttributeType
        sCharacterId.isOptional = false

        let sCreatedAt = NSAttributeDescription()
        sCreatedAt.name = "createdAt"
        sCreatedAt.attributeType = .dateAttributeType
        sCreatedAt.isOptional = false

        let sUpdatedAt = NSAttributeDescription()
        sUpdatedAt.name = "updatedAt"
        sUpdatedAt.attributeType = .dateAttributeType
        sUpdatedAt.isOptional = false

        session.properties = [sId, sTitle, sCharacterId, sCreatedAt, sUpdatedAt]

        // Message
        let message = NSEntityDescription()
        message.name = "CDMessage"
        message.managedObjectClassName = "NSManagedObject"

        let mId = NSAttributeDescription()
        mId.name = "id"
        mId.attributeType = .UUIDAttributeType
        mId.isOptional = false

        let mSessionId = NSAttributeDescription()
        mSessionId.name = "sessionId"
        mSessionId.attributeType = .UUIDAttributeType
        mSessionId.isOptional = false

        let mRole = NSAttributeDescription()
        mRole.name = "role"
        mRole.attributeType = .stringAttributeType
        mRole.isOptional = false

        let mContent = NSAttributeDescription()
        mContent.name = "content"
        mContent.attributeType = .stringAttributeType
        mContent.isOptional = false

        let mCreatedAt = NSAttributeDescription()
        mCreatedAt.name = "createdAt"
        mCreatedAt.attributeType = .dateAttributeType
        mCreatedAt.isOptional = false

        message.properties = [mId, mSessionId, mRole, mContent, mCreatedAt]

        // Memory
        let memory = NSEntityDescription()
        memory.name = "CDMemory"
        memory.managedObjectClassName = "NSManagedObject"

        let memId = NSAttributeDescription()
        memId.name = "id"
        memId.attributeType = .UUIDAttributeType
        memId.isOptional = false

        let memSessionId = NSAttributeDescription()
        memSessionId.name = "sessionId"
        memSessionId.attributeType = .UUIDAttributeType
        memSessionId.isOptional = false

        let memText = NSAttributeDescription()
        memText.name = "text"
        memText.attributeType = .stringAttributeType
        memText.isOptional = false

        let memScore = NSAttributeDescription()
        memScore.name = "score"
        memScore.attributeType = .doubleAttributeType
        memScore.isOptional = false

        let memCreatedAt = NSAttributeDescription()
        memCreatedAt.name = "createdAt"
        memCreatedAt.attributeType = .dateAttributeType
        memCreatedAt.isOptional = false

        memory.properties = [memId, memSessionId, memText, memScore, memCreatedAt]

        // Character
        let character = NSEntityDescription()
        character.name = "CDCharacter"
        character.managedObjectClassName = "NSManagedObject"

        let cId = NSAttributeDescription()
        cId.name = "id"
        cId.attributeType = .stringAttributeType
        cId.isOptional = false

        let cName = NSAttributeDescription()
        cName.name = "name"
        cName.attributeType = .stringAttributeType
        cName.isOptional = false

        let cPersonality = NSAttributeDescription()
        cPersonality.name = "personality"
        cPersonality.attributeType = .stringAttributeType
        cPersonality.isOptional = false

        let cWorld = NSAttributeDescription()
        cWorld.name = "world"
        cWorld.attributeType = .stringAttributeType
        cWorld.isOptional = false

        let cTone = NSAttributeDescription()
        cTone.name = "tone"
        cTone.attributeType = .stringAttributeType
        cTone.isOptional = false

        character.properties = [cId, cName, cPersonality, cWorld, cTone]

        model.entities = [session, message, memory, character]
        return model
    }
}

