class InsufficientDataError(Exception):

    def __init__(self, actual_size, expected_size, message=None):
        if message is None:
            message = 'The input data is insufficient for the column type heuristics to work. ' \
                      'Only {actual} row(s) were passed. Please pass at least {expected} rows.'\
                      .format(actual=actual_size, expected=expected_size)
        super().__init__(message)
        self.actual_size = actual_size
        self.expected_size = expected_size


class UnknownMetadataReturnColumnTypeError(Exception):

    def __init__(self, mdtype, message=None):
        if message is None:
            message = 'Return column type {type} of {metadata} is unknown. Should be numerical or categorical.'\
                      .format(type=mdtype.metadata_return_type(), metadata=mdtype)
        super().__init__(message)
        self.mdtype = mdtype

