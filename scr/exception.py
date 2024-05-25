import sys
from scr.logger import logging
def error_message_detail(error,error_detail:sys):
    _,_,error_tab = error_detail.exc_info() 
    file_name = error_tab.tb_frame.tb_code.co_filename
    error_message = "Error occured in pyhton script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,error_tab.tb_lineno,str(error)
    )
    return error_message



class CutomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super.__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_details)

    def __str__(self):
        return self.error_message