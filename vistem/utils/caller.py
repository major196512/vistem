import sys
import os

__all__ = ['find_caller']

def find_caller():
    frame = sys._getframe(0)
    while frame:
        code = frame.f_code
        caller = frame.f_globals['__name__']
        if caller in ['__main__', '__mp_main__'] : caller = 'vistem'
        
        if (os.path.join("utils", "caller.") not in code.co_filename) and (os.path.join("utils", "logger.") not in code.co_filename):
            return {
                'caller' : caller,
                'file_name' : code.co_filename,
                'line_num' : frame.f_lineno,
                'func_name' : code.co_name
            }
            
        frame = frame.f_back
