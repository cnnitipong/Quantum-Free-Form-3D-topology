"""Exception types of the FreeTO core."""


class FreeTOError(ValueError):
    """A problem definition that FreeTO cannot solve (the message is meant for
    the end user).  Subclass of ValueError, so ``except ValueError`` catches
    it as well."""


SINGULAR_MSG = ("The stiffness matrix is singular: the supports do not prevent "
                "rigid-body motion of the structure. Add or enlarge fixed / "
                "xfixed / yfixed / zfixed regions so that they overlap the "
                "domain and restrain all translations and rotations.")
