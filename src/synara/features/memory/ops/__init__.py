"""High-level memory operations exposed via ``MemoryService`` delegates.

Each module here defines an ``async def run(service, *, ...)`` entry
point that the service calls. Keeping them in a sibling subpackage
isolates the operation surface from the smaller neural primitives in
``..primitives`` and from the service core itself.
"""
