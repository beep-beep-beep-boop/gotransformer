namespace Gotransformer

open TorchSharp
open Argu
open Goban

[<CliPrefix(CliPrefix.DoubleDash)>]
type TrainArgs =
    | [<Mandatory>] Train_data of path: string
    | [<Mandatory>] Validation_data of path: string
    | [<Mandatory>] Test_data of path: string
    | Model of path: string

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Train_data _ -> "path to the train data json file"
            | Validation_data _ -> "path to the validation data file"
            | Test_data _ -> "path to the test data json file"
            | Model _ -> "path to the model to restore and continue training from (optional)"

and TestArgs =
    | [<Hidden>] Hidden

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Hidden -> ""

and GotransformerArgs =
    | [<CliPrefix(CliPrefix.None)>] Train of ParseResults<TrainArgs>
    | [<CliPrefix(CliPrefix.None)>] Test of ParseResults<TestArgs>

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Train _ -> "Train the model"
            | Test _ -> "execute a special recipie"



module Main =
    let train train_path val_path test_path (model_path:Option<string>)=
        printfn "loading data..."
        let d_train = Dataset.as_tensor (Dataset.load train_path)
        let d_val = Dataset.as_tensor (Dataset.load val_path)
        let d_test = Dataset.as_tensor (Dataset.load test_path)

        let model = match model_path with
                    | Some path ->
                        printfn "restoring model..."
                        let m = new Transformer.TransformerModel(Token.vocab_size, Transformer.device)
                        m.load(path) |> ignore
                        Some m
                    | None -> None

        Transformer.run d_train d_val d_test 1 model |> ignore

        0
    

    let cmd_test _ =
        printfn "loading model..."
        let device = torch.CPU
        let model = new Transformer.TransformerModel(Token.vocab_size, device)
        model.load("model.dat") |> ignore

        printfn "predicting..."
        let move1 = {Col=3; Row=6; Color=Black}
        let input = [| Token.token_start ; Token.move_to_token move1; |]
        Transformer.predict model input |> Token.token_to_move |> printfn "%A"

        // now how do we make a prediction from this :\
        0


    [<EntryPoint>]
    let main argv =
        let parser = ArgumentParser.Create<GotransformerArgs>(programName = "gotransformer")

        let result =
            try
                Some(parser.ParseCommandLine(inputs = argv, raiseOnUsage = true))
            with
            | e ->
                printfn "%s" e.Message
                None

        match result with
        | None -> -1
        | Some args ->
            if args.Contains Train then
                // ugh this shitty library sucks...
                let targs = args.GetResult Train

                if
                    (not (targs.Contains Train_data))
                    || (not (targs.Contains Test_data))
                    || (not (targs.Contains Validation_data))
                then
                    printfn "yo ur missing something in the train subcommand args"
                    printfn "run with --help to know what u need"
                    -1
                else
                    let model = if targs.Contains Model then Some (targs.GetResult Model) else None
                    train (targs.GetResult Train_data) (targs.GetResult Validation_data) (targs.GetResult Test_data) model
            else if args.Contains Test then
                cmd_test ()
            else
                printfn "%s" (parser.PrintUsage())
                -1



// code for loading the model
//let model = new Transformer.TransformerModel(Conv.vocab_size, torch.CPU)
//model.load("model.dat") |> ignore
//
//let d_val = Dataset.as_tensor (Dataset.load "val.json")
//let test_data = Transformer.batchify d_val 64L torch.CPU
//let tst_loss = Transformer.evaluate model test_data Conv.vocab_size
//
//printfn "%A" tst_loss
