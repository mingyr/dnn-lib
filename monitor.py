import os
import sys
import re
import subprocess
import signal
import string


class Parser(object):
    def __init__(self):
        pass

    def get_start_parser(self):
        reg = re.compile("start training", re.DOTALL)
        def __parser(cont):
            output = reg.search(cont)
            if output != None:
                return output.group(0)
            else:
                return None
        return __parser

    def get_finish_parser(self):
        def __parser(cont):
            reg = re.compile("training finished", re.DOTALL)
            output = reg.search(cont)
            if output != None:
                return output.group(0)
            else:
                return None
        return __parser

    def get_trim_newline_parser(self):
        def __parser(cont):
            return string.strip(cont, os.linesep)
        return __parser

    def get_custom_parser(self, criteria):
        def __parser(cont):
            reg = re.compile(criteria, re.DOTALL)
            output = reg.search(cont)
            if output != None:
                return output.group(0)
        return __parser

    def __del__(self):
        pass

class Cmd:
    def __init__(self, parser = None, results = None, cwd = None):
        self._cwd = (None if cwd == '' else cwd)
        self._running = False
        self._parser = parser
        self._results = results
        signal.signal(signal.SIGCHLD, self._sig_handler())
        
    def start(self, cmdline, output_file = None):
        if output_file:
            self._reg_file = True
            self._output_file = output_file
            with open(output_file, "w") as f:
                self._child = subprocess.Popen(cmdline, stdout = f, cwd = self._cwd)
        else:
            self._reg_file = False
            self._child = subprocess.Popen(cmdline, stdout = sys.stdout, cwd = self._cwd)

        self._running = True
        return 0

    def _sig_handler(self):
        def handler(sig, frame):
            print("SIGCHLD received")
            self._running = False
            if self._parser:
                output = self.parse(self._parser)
                if output:
                    print(output)
                    if type(self._results) == list:
                        self._results.append(output)
            
        return handler
        
    def stop(self):
        try:
            self._child.send_signal(signal.SIGINT)         
        except AttributeError:
            return -1
        else:       
            self._running = False
            return 0
            
    def parse(self, parser):
        if self._reg_file:
            with open(self._output_file, "r") as g:
                cont = "".join(g.readlines())

            return parser(cont) if cont else None
        else:
            return None

    def wait(self):
        self._child.wait()

    def __del__(self):
        if self._running == True:
            signal.signal(signal.SIGCHLD, signal.SIG_DFL)
            self.stop()
    
def test():
    import unittest
    import time

    class CmdTest(unittest.TestCase):
        def test_parse(self):
            def parse(cont):
                print(cont)
                reg = re.compile("start training")
                output = reg.search(cont)
                if output != None:
                    return output.group(0)
                else:
                    return "Not found"

            test_cmd = Cmd()
            test_cmd.start(['python', 'test.py'], "tmp")
            time.sleep(1)
            print(test_cmd.parse(parse))
            self.assertEqual(0, 0)
        
    cmdtest_suite = unittest.TestSuite()
    for test_name in ['test_parse']:        
        cmdtest_suite.addTest(CmdTest(test_name))
    
    unittest.TextTestRunner(verbosity = 2).run(cmdtest_suite)

if __name__ == '__main__':
    test()
    
