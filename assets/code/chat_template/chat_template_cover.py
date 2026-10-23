"""Cover for the chat template post: the messages you send next to the string the model reads.
The right-hand text is the exact output of section [1] in chat_template_out.txt."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, MUTED, TEAL, PINK, ORANGE = "#1e1f26", "#e8e8ef", "#9a9ab0", "#4fb3bf", "#e06c9f", "#f2a65a"
W, H = 1200, 630
out = open("chat_template_out.txt").read()
rendered = out.split("(no system message given)\n", 1)[1].split("\n[2]", 1)[0].rstrip("\n").split("\n")
# wrap the long user line for display only
rendered = [l2 for l in rendered for l2 in (l.replace("? Answer", "?\nAnswer").split("\n"))]

fig = plt.figure(figsize=(12, 6.3), facecolor=BG)
put = lambda x, y, s, **kw: fig.text(x / W, 1 - y / H, s, ha="left", va="center", **kw)
put(80, 76, "Chat templates: what the model actually reads", fontsize=24, fontweight="bold", color=FG)

put(80, 160, "you send", fontsize=15, color=MUTED)
put(80, 205, '[{"role": "user",', fontsize=15, family="monospace", color=FG)
put(80, 237, '  "content": "What is the', fontsize=15, family="monospace", color=FG)
put(80, 269, '   capital of Australia?', fontsize=15, family="monospace", color=FG)
put(80, 301, '   Answer in one sentence."}]', fontsize=15, family="monospace", color=FG)
put(420, 253, "→", fontsize=34, color=ORANGE)

put(480, 160, "the model reads (Qwen2.5 template)", fontsize=15, color=MUTED)
r = fig.canvas.get_renderer()
y = 205
for line in rendered:
    x = 480
    # split around the two control tokens so they can be coloured
    parts = line.replace("<|im_start|>", "\0<|im_start|>\0").replace("<|im_end|>", "\0<|im_end|>\0").split("\0")
    for part in filter(None, parts):
        if part.startswith("<|"):
            color = PINK
        elif part == "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.":
            part, color = "You are Qwen, created by Alibaba Cloud.", ORANGE   # shortened to fit; full text in the post
        else:
            color = FG
        t = put(x, y, part, fontsize=13, family="monospace", color=color)
        x += t.get_window_extent(r).width
    y += 32
put(480, y + 22, "pink: control tokens   orange: default system prompt", fontsize=13, color=MUTED)
put(80, 540, "Same question without the template: the model kept going and added wrong facts.", fontsize=15, color=FG)
fig.savefig("chat_template_cover.png", dpi=100)
