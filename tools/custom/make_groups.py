num_hidden_layers = 12
group_size = 3
keep_last = -1

idx = []
for i in range(group_size, num_hidden_layers + 1, group_size):
    idx.extend([i + kl for kl in range(keep_last, 0)])

print(idx)