namespace Goban

[<AutoOpen>]
module Types =
    type Color =
        | White
        | Black

    type Move = { Color: Color; Row: int; Col: int }
