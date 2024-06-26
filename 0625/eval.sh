for i in 1 2; do
    steps=$((i * 62500))
    echo "steps: $steps"
    
    python generator_rev.py --model "heegyu/0625-gemma-2B-infini-qarv" --revision "lr5e-6-steps-$steps" --gpu_devices 0 --template "templates/template-gemma.json" --output_dir "0625/"
    python judgement.py -o "0625/heegyu_0625-gemma-2B-infini-qarv@lr5e-6-steps-$steps.jsonl"
done
