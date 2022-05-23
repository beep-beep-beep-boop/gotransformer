﻿namespace Gotransformer

open TorchSharp
open Argu
open Goban

[<CliPrefix(CliPrefix.DoubleDash)>]
type TrainArgs =
    | [<Mandatory>] Train_data of path: string
    | [<Mandatory>] Validation_data of path: string
    | [<Mandatory>] Test_data of path: string

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Train_data _ -> "path to the train data json file"
            | Validation_data _ -> "path to the validation data file"
            | Test_data _ -> "path to the test data json file"

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
    let train train_path val_path test_path =
        printfn "loading data..."
        let d_train = Dataset.as_tensor (Dataset.load train_path)
        let d_val = Dataset.as_tensor (Dataset.load val_path)
        let d_test = Dataset.as_tensor (Dataset.load test_path)

        Transformer.run d_train d_val d_test 1 |> ignore

        0
    
    let predict (model:Transformer.TransformerModel) (input:Token array) =
        model.eval ()

        let mutable input_tensor = Token.array_to_tensor input

        let mask = model.GenerateSquareSubsequentMask (input_tensor.shape.[0])
        use output = model.forward (input_tensor, mask)

        let struct (a, b) = (output.topk 1)
        let next_item:int64= b.view(-1).[-1].item()

        let next_item_32 = int32 next_item
        if (int64 next_item_32) <> next_item then
            raise (System.Exception("the int64 was too big ???"))

        Token.token_to_move next_item_32

    let cmd_test _ =
        printfn "loading model..."
        let device = torch.CPU
        let model = new Transformer.TransformerModel(Token.vocab_size, device)
        model.load("model.dat") |> ignore

        printfn "predicting..."
        let move1 = {Col=3; Row=6; Color=Black}
        let input = [| Token.token_start ; Token.move_to_token move1; |]
        predict model input |> printfn "%A"

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
                    train (targs.GetResult Train_data) (targs.GetResult Validation_data) (targs.GetResult Test_data)
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
