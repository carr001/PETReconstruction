import sys
sys.path.append('..\\Models')

import moduletest as mt
import submodule as sm # can not recognize subsubmodule
import submodule.submoduletest as smt #you cant import a dir as a module, such as import submodule as smt, smt will have no attribute named submoduletest

print(mt.add(1,2))
print(smt.sub(10,2))



