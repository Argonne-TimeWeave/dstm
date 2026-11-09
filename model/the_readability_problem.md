# The Readability Problem

## The basic issue

DSTM allows more than one operation to have the same local time. A rank may therefore still have another write at time `t` even after the runtime has observed that its current time is `t`.

This matters when deciding whether version `t` is readable. A version is readable only when nobody can write to it anymore. Knowing only that the owner's clock is `t` does not prove this: another local write at `t` may still be issued.

The two supported program classes make the distinction important:

- A **push program** writes to remote future versions and reads its local present version.
- A **pull program** writes its local present version and reads remote past versions.

For the current deadline, these are separate classes. A valid program is globally push-only or globally pull-only; it does not mix the two classes. They should be proved separately where their progress arguments differ.

Separating the classes helps the proofs, but it does not by itself establish runtime readability. The runtime must still know that every write which could target a version is finished.

## Why the current clock test is not enough

The intended rule says that the local clock must move strictly past the version time before the version becomes readable (`context.tex:397-406`). The runtime comment says the same thing, but its current check accepts equality (`../src/ops.c:114-124`).

Remote writes also matter. A later heartbeat is useful only if receiving it proves that all earlier writes from that rank were delivered and applied. The current message and barrier behavior does not yet provide a sufficiently clear ordering or closure guarantee; these issues are recorded in `CONCERNS.md`.

## Suggested long-term fix

Add explicit end-of-writes evidence:

1. A rank announces that it will issue no more writes originating at or before time `t`.
2. The runtime ensures the affected writes have been delivered and applied before other ranks use that announcement.
3. A version becomes readable only after its owner and every relevant neighbor provide enough closure evidence.
4. The barrier carries and validates its generation, phase, and agreed time frontier before it acts as global closure evidence.

This would support the final operation time even when no later ordinary operation exists. It would also avoid treating a heartbeat as proof of message delivery when the runtime has not established that ordering.

Changing the runtime is outside the current deadline scope. For now, the Rocq model keeps the sound strict readability condition, treats push and pull as separate valid program classes, and proves only results supported by explicit evidence. `abstract_step_lifts_with_evidence` shows that an abstract step can be reproduced once the required readability or discardability evidence is available. It does not claim that the current runtime always produces that evidence.
