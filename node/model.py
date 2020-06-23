import torch
import torch.nn as nn
import string
import random
import sys
import unidecode

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get characters from string.printable
all_characters = string.printable
n_characters = len(all_characters)


# Parses the file to remove all non-alphabetic characters

# with open(r"C:\Users\Adams\PycharmProjects\TextRNN\Nietzsche.txt", encoding='utf-8') as f, open("cleaned_file.txt", "w") as n:
#  x = f.read()
# result = re.sub(r"[^a-z\s]", "", x, 0, re.IGNORECASE | re.MULTILINE)
# n.write(result)


# Reads the cleaned text file
file = unidecode.unidecode(open(r"C:\Users\Adams\PycharmProjects\TextRNN\cleaned_file.txt", encoding='utf-8').read())


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        out = self.embed(x)
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell


class Generator:
    def __init__(self):
        self.chunk_len = 250
        self.num_epochs = 20000
        self.batch_size = 1
        self.print_every = 50
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.003
        self.rnn = RNN(n_characters, self.hidden_size, self.num_layers, n_characters).to(device)

    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_characters.index(string[c])
        return tensor

    def get_random_batch(self):
        start_idx = random.randint(0, len(file) - self.chunk_len)
        end_idx = start_idx + self.chunk_len + 1
        text_str = file[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_str[:-1])
            text_target[i, :] = self.char_tensor(text_str[1:])

        return text_input.long(), text_target.long()

    lower_upper_alphabet = string.ascii_letters.upper()
    random_letter = random.choice(lower_upper_alphabet)

    def generate(self, initial_str=str(random_letter), predict_len=100, temperature=0.85):

        hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        for p in range(len(initial_str) - 1):
            _, (hidden, cell) = self.rnn(
                initial_input[p].view(1).to(device), hidden, cell
            )

        last_char = initial_input[-1]

        for p in range(predict_len):
            output, (hidden, cell) = self.rnn(
                last_char.view(1).to(device), hidden, cell
            )
            output_dist = output.data.view(-1).div(temperature).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predicted_char = all_characters[top_char]
            predicted += predicted_char
            last_char = self.char_tensor(predicted_char)

        return predicted

    # input_size, hidden_size, num_layers, output_size
    def train(self):

        # View RNNs parameters
        print("Model's state_dict:")
        for param_tensor in self.rnn.state_dict():
            print(param_tensor, "\t", self.rnn.state_dict()[param_tensor].size())

        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        print("=> Starting training")

        for epoch in range(1, self.num_epochs + 1):
            inp, target = self.get_random_batch()
            hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)

            self.rnn.zero_grad()
            loss = 0
            inp = inp.to(device)
            target = target.to(device)

            for c in range(self.chunk_len):
                output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell)
                loss += criterion(output, target[:, c])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_len

            # Saves the model
            path = r"C:\Users\Adams\PycharmProjects\TextRNN\parameters.pth"
            torch.save(self.rnn.state_dict(), path)

            print("Epoch: ", epoch)
            print(f"Loss: {loss}")

            print(self.generate())

    def load_model(self):
        trained = self.rnn
        trained.load_state_dict(torch.load(r"C:\Users\Adams\PycharmProjects\TextRNN\parameters.pth"))
        trained.eval()


gennames = Generator()
# gennames.train()
gennames.load_model()
print(gennames.generate())



