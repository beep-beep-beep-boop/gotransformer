namespace Gotransformer

open Goban
open TorchSharp

module Token =
    let special_tokens = 2
    let token_start: Token = 0
    let token_end: Token = 1

    let board_size = 19
    let total_intersections = board_size * board_size
    let vocab_size = (total_intersections * 2) + special_tokens

    let move_to_token move : Token =
        let r = move.Row - 1
        let c = move.Col - 1

        (r * board_size)
        + c
        + (if move.Color = White then
               total_intersections
           else
               0)
        + special_tokens

    let token_to_move (token: Token) =
        let n = token - special_tokens

        let (color, n) =
            if n / total_intersections = 1 then
                (White, n - total_intersections)
            else
                (Black, n)

        let r = n / board_size
        let c = n % board_size

        { Row = r + 1
          Col = c + 1
          Color = color }
    
    let array_to_tensor (x: Token array) = torch.tensor (Array.map int64 x)
