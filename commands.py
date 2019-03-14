import sys

def commands():
    if len(sys.argv) > 2:
       filecode,typeof,folder=sys.argv
       if typeof == "--fit":
           """Executer le programme d'entrainement"""
       elif typeof == "--predict":
           """Executer le programme de prediction"""
       else:
           usage(filecode)
    else:
        usage(filecode)
     
def usage(filecode):
    print(filecode+": --fit <folder>")
    print(filecode+": --predict <folder>")
            
commands()