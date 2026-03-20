class RoleplaySystemError(Exception):
    """Base error for the roleplay system."""


class ConfigError(RoleplaySystemError):
    pass


class DatasetError(RoleplaySystemError):
    pass


class CharacterProfileError(RoleplaySystemError):
    pass


class PromptBuilderError(RoleplaySystemError):
    pass


class MemoryStoreError(RoleplaySystemError):
    pass

