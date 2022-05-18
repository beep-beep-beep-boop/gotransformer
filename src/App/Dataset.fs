namespace Gotransformer

open FSharp.Json

module Dataset =
    let load path =
        let text = System.IO.File.ReadAllText path
        Json.deserialize<Dataset> text
