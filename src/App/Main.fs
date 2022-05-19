namespace Gotransformer

module Main =

    [<EntryPoint>]
    let main _ =

        printfn "loading data..."
        let d_train = Dataset.as_tensor (Dataset.load "train.json")
        let d_val = Dataset.as_tensor (Dataset.load "val.json")

        do Transformer.run d_train d_val d_val 1 |> ignore

        0
