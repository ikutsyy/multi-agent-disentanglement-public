import pickle
import subprocess
import codecs
import base64

def start(executable_file):
    return subprocess.Popen(
        executable_file,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE)


def read(process):
    p = process.stdout.readline().decode("utf-8").strip()
    print(p[2:-3].replace("\\n",""))
    return pickle.loads(codecs.decode(p[2:-3].replace("\\n","").encode(), "base64"))


def write(process):
    obj = ("hello "*1000,5)
    pickled = codecs.encode(pickle.dumps(obj), "base64")
    print("Sending",pickled)
    process.stdin.write(pickled)
    process.stdin.flush()


def terminate(process):
    process.stdin.close()
    process.terminate()
    process.wait(timeout=0.2)


process = start("experiment_shell.sh")
write(process)
print("read",read(process))
write(process)
print("read",read(process))
terminate(process)