namespace Gotransformer

open FSharp.Json
open TorchSharp

module Dataset =
    let load path =
        let text = System.IO.File.ReadAllText path
        Json.deserialize<Dataset> text

    let as_tensor (dataset: Dataset) =
        let toTensor (x: Token array) = torch.tensor (Array.map int64 x)
        torch.cat ((Array.map toTensor dataset), 0)
