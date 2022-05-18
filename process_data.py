from sgfmill import sgf
from sgfmill import sgf_moves
import glob
import enum
from collections import namedtuple
import random
import json

special_tokens = 3
token_start = 0
token_end = 1
token_pad = 2

board_size = 19

# tag::color[]
class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def other(self):
        return Player.black if self == Player.white else Player.white
# end::color[]

# tag::points[]
class Point(namedtuple('Point', 'row col')):
    def neighbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]
# end::points[]

total_intersections = board_size * board_size

vocab_size = (total_intersections * 2) + special_tokens

def move_to_number(color, row, col):
    r = row - 1
    c = col - 1

    n = (r * board_size)
    n += c

    if color == Player.white:
        n += total_intersections
    
    n += special_tokens

    return n

def number_to_move(n):
    n -= special_tokens

    color = Player.black
    if (n // total_intersections) == 1:
        n -= total_intersections
        color = Player.white
    
    r = n // board_size
    c = n % board_size

    row = r + 1
    col = c + 1

    return color, row, col


def player_from_color(color):
    if color == 'w':
        return Player.white
    elif color == 'b':
        return Player.black
    else:
        raise Exception('uhhhh thats not a color')



def process_game(input_path):
    f = open(input_path, 'rb')
    sgf_src = f.read()
    f.close()

    try:
        sgf_game = sgf.Sgf_game.from_bytes(sgf_src)
    except ValueError:
        raise Exception('bad sgf file')

    b_size = sgf_game.get_size()

    try:
        board, plays = sgf_moves.get_setup_and_moves(sgf_game)
    except ValueError as e:
        raise Exception(str(e))

    try:
        white_rank = sgf_game.root.get('WR')
        black_rank = sgf_game.root.get('BR')
    except:
        white_rank = None
        black_rank = None

    try:
        handicap = sgf_game.get_handicap()
    except ValueError:
        raise Exception('handicap information malformed')

    if handicap == None:
        handicap = 0

    game_data = [token_start]

    for color, move in plays:
        if move is None:
            continue
        row, col = move

        game_data.append(move_to_number(player_from_color(color), row + 1, col + 1))

        # make sure its not FUCKED
        #try:
        #    board.play(row, col, color)
        #except ValueError:
        #    raise Exception('illegal move in sgf file')

    game_data.append(token_end)

    return b_size, handicap, white_rank, black_rank, game_data

if __name__ == "__main__":
    print("Warning - make sure the constants in this script are set to what you're using in the other code (ie, board size, pad token, etc)")

    paths = glob.glob('./games/**/*.sgf', recursive=True)
    random.shuffle(paths)
    max_game_amount = len(paths)
    max_game_length = 512

    # amount of data to put in the test dataset
    # should be between 0 and 1
    split_test_percent = 0.15

    # note - these might not be all filled up, these are just the max amounts
    # (at least the train one)
    test_amount = int(max_game_amount * split_test_percent)
    train_max_amount = max_game_amount - test_amount

    d_train = []
    d_val = []

    # TODO: we might just want to throw everything into the same file first and then split it after,
    #   once we know how much valid data there is

    num = 0
    for path in paths:
        try:
            b_size, handicap, wr, br, game = process_game(path)
        except:
            print('err')
            continue

        if handicap != 0:
            continue

        if b_size != board_size:
            continue

        if num < test_amount:
            d_val.append(game)
        else:
            d_train.append(game)

        num += 1

        if num % 1_000 == 0:
            print(f'{num}/{max_game_amount}')

    print(f'{num+1}/{max_game_amount} games valid. ({max_game_amount - (num+1)} invalid)')

    print('writing data to disk...')

    with open('train.json', 'w') as f:
        json.dump(d_train, f)
    
    with open('val.json', 'w') as f:
        json.dump(d_val, f)
