namespace Gotransformer

open FSharp.Json
open TorchSharp

module Dataset =
    let load path =
        let text = System.IO.File.ReadAllText path
        Json.deserialize<Dataset> text

    let as_tensor (dataset: Dataset) =
        torch.cat ((Array.map Token.array_to_tensor dataset), 0)
