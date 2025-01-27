#(pyimagej_env) C:\Users\rodri\Doutorado_2023\detect_lesions>mamba activate pyimagej

#(pyimagej) C:\Users\rodri\Doutorado_2023\detect_lesions>

#https://pyimagej.readthedocs.io/en/latest/06-Working-with-Images.html

import imagej


ij = imagej.init()
imp = ij.io().open("1-000-P_00557_08.jpg")
ij.py.show(imp)
ij.op().run(imp, "Green Fire Blue", "")