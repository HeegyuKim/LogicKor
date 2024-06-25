for i in 1; do
    steps=$((i * 100000))
    python generator_rev.py --model "heegyu/0620-qwen2-7B-infini-qarv" --revision "lr2e-5-steps-$steps" --gpu_devices 2 --template "templates/template-gemma.json" --output_dir "0620/"
    python judgement.py -o "0620/heegyu_0620-qwen2-7B-infini-qarv@lr2e-5-steps-$steps.jsonl"
done
