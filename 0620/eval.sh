for i in 3 4 5; do
    steps=$((i * 100000))
    echo "steps: $steps"
    
    # python generator_rev.py --model "heegyu/0620-gemma-2B-infini-qarv" --revision "lr2e-5-steps-$steps" --gpu_devices 3 --template "templates/template-gemma.json" --output_dir "0620/"
    python judgement.py -o "0620/heegyu_0620-gemma-2B-infini-qarv@lr2e-5-steps-$steps.jsonl"
done
