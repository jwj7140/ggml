import subprocess
import shlex
import os

class ggmlTextModel:
    def __init__(self):
        self.model_path = ""
        self.thread= 8
        self.batch_size= 8
        self.prompt = ""
        self.tokens = []
        self.model_info = 0
    
    def setting(self, model_path="", thread=8, batch_size=8, model_type="stablelm"):
        self.model_path=model_path
        self.thread=thread
        self.batch_size=batch_size
        self.model_type=model_type
        self.path = "/".join(os.path.realpath(__file__).split("/")[0:-1])+"/build/bin/"+self.model_type
    
    def info(self):
        self.command = [
            self.path,
            "-m",
            self.model_path,
            "-n",
            str(0),
            "-p",
            "."
        ]
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        self.model_info = {}
        line_count = 0
        while self.process.stdout.readable():
            line = self.process.stdout.readline()
            if (line_count > 9 or not line):
                self.quit()
                break
            if (line.find("n_vocab") != -1):
                self.model_info["n_vocab"] = int(line.split("=")[1])
            if (line.find("n_ctx") != -1):
                self.model_info["n_ctx"] = int(line.split("=")[1])
            if (line.find("n_embd") != -1):
                self.model_info["n_embd"] = int(line.split("=")[1])
            if (line.find("n_head") != -1):
                self.model_info["n_head"] = int(line.split("=")[1])
            if (line.find("n_layer") != -1):
                self.model_info["n_layer"] = int(line.split("=")[1])
            if (line.find("n_rot") != -1):
                self.model_info["n_rot"] = int(line.split("=")[1])
            if (line.find("ftype") != -1):
                self.model_info["ftype"] = int(line.split("=")[1])
            line_count = line_count + 1
        return self.model_info
    
    #tokenize
    def encode(self, prompt):
        if (self.prompt == prompt):
            return self.tokens

        self.prompt = prompt
        # self.prompt = self.prompt.replace("\n", "\\n")
        # self.prompt = self.prompt.replace("\"", "\\\"")
        self.command = [
            self.path,
            "-m",
            self.model_path,
            "-n",
            str(0),
            "-p",
            self.prompt
        ]
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        token_count = 0
        token_num = 0
        self.tokens = []
        while self.process.stdout.readable():
            line = self.process.stdout.readline()
            if (len(line) < 13 or not line):
                self.quit()
                break
            if (token_count > 0 and line.find("token") != -1):
                token_count = token_count - 1
                self.tokens.append(int(line.split("=")[1].split(",")[0])) 
            if (line.find("number of tokens in prompt") != -1):
                token_count = int(line.split("=")[1])
                token_num = int(line.split("=")[1])
        return range(token_num)

    def generate(self, n_predict=100, top_p=1, top_k=50, temperature=0.9, seed=-1, repeat_penalty=1.2, prompt=""):
        self.prompt = prompt
        # self.prompt = self.prompt.replace("\n", "\\n")
        # self.prompt = self.prompt.replace("\"", "\\\"")
        self.command = [
            self.path,
            "-m",
            self.model_path,
            "-t",
            str(self.thread),
            "-n",
            str(n_predict),
            "--temp",
            str(temperature),
            "-s",
            str(seed),
            "--top_k",
            str(top_k),
            "--top_p",
            str(top_p),
            "-b",
            str(self.batch_size),
            "--repeat_penalty",
            str(repeat_penalty),
            "--split",
            "-p",
            self.prompt
        ]
        # print(" ".join(self.command))
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
        reply = ""
        token_count = 0
        self.tokens = []
        while self.process.stdout.readable():
            line = self.process.stdout.readline()
            if not line:
                self.quit()
                break
            if (1 < len(line) and len(line) < 13):
                if (int(line[:-1].split(":", 1)[0]) == 202):
                    #new line token find
                    text = "\n"
                else:
                    text = line[:-1].split(":", 1)[1]
                reply = reply + text
                if (len(reply) > len(self.prompt)):
                    yield text
            else:
                if (token_count > 0 and line.find("token") != -1):
                    token_count = token_count - 1
                    self.tokens.append(int(line.split("=")[1].split(",")[0])) 
                if (line.find("number of tokens in prompt") != -1):
                    token_count = int(line.split("=")[1])
        # print(self.tokens)
        # print(reply)
        return 0

    def quit(self):
        self.process.kill()




# prompt = """그는"""

# model = ggmlTextModel()
# model.setting("./polyglot-12.8B/ggml-polyglot-ko-12.8B-q4_2.bin", 16, 256, "stablelm")
# print(model.info())
# print(model.encode(prompt))

# for text in model.generate(100, 1, 50, 0.9, -1, 1.2, prompt):
#     print(text, end="", flush=True)
