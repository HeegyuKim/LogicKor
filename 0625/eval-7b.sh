for i in 1 2; do
    steps=$((i * 62500))
    python generator_rev.py --model "heegyu/0625-qwen2-7B-infini-qarv" --revision "lr5e-6-steps-$steps" --gpu_devices 2 --template "templates/template-gemma.json" --output_dir "0625/"
    python judgement.py -o "heegyu_0625-qwen2-7B-infini-qarv@lr5e-6-steps-$steps.jsonl"
done
