namespace Gotransformer

module Main =

    [<EntryPoint>]
    let main _ =

        printfn "loading data..."
        let d_train = Dataset.load"train.json"
        let d_val = Dataset.load "val.json"

        0
