
for epoch in 2 3; do
    # python generator_rev.py --model "heegyu/0613-Qwen2-7B-en-kolima" --revision epoch-$epoch --gpu_devices 3 --template "templates/template-chatml.json" --output_dir "0613/"
    # python judgement.py -o "0613/heegyu_0613-Qwen2-7B-en-kolima@epoch-$epoch.jsonl"
    
    # heegyu/0613-Qwen2-7B-en
    # python generator_rev.py --model "heegyu/0613-Qwen2-7B-en" --revision epoch-$epoch --gpu_devices 3 --template "templates/template-chatml.json" --output_dir "0613/"
    # python judgement.py -o "0613/heegyu_0613-Qwen2-7B-en@epoch-$epoch.jsonl"

    # heegyu/0613-Qwen2-7B-en-kolima-qarv
    # python generator_rev.py --model "heegyu/0613-Qwen2-7B-en-kolima-qarv" --revision epoch-$epoch --gpu_devices 2 --template "templates/template-chatml.json" --output_dir "0613/"
    # python generator_hf_chat.py --model "heegyu/0613-Qwen2-7B-en-kolima-qarv" --revision epoch-$epoch --output_dir "0613/"
    python judgement.py -o "0613/heegyu_0613-Qwen2-7B-en-kolima-qarv@epoch-$epoch.jsonl"
done

# for epoch in 1 2 3; do
#     python score.py -p "0613/heegyu_0613-Qwen2-7B-en-kolima@epoch-$epoch.jsonl"
# done