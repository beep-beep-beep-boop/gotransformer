namespace Gotransformer

open Goban

[<AutoOpen>]
module Types =
    type Token = int

    type Dataset = (Token array) array