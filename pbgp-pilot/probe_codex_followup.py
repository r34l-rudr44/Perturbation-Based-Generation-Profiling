import json, pathlib, urllib.request, urllib.error

auth = json.loads(pathlib.Path('C:/Users/nevrohelios/.codex/auth.json').read_text(encoding='utf-8'))
tokens = auth['tokens']
headers = {'Authorization': 'Bearer ' + tokens['access_token'],
           'ChatGPT-Account-Id': tokens['account_id'],
           'Content-Type': 'application/json', 'Accept': 'text/event-stream'}
base = {'model': 'gpt-6.1-sol', 'instructions': 'Reply briefly.',
        'input': [{'role':'user','content':[{'type':'input_text','text':'Reply with exactly: hello'}]}],
        'store': False, 'stream': True, 'tools': [], 'reasoning': {'effort':'low'}}
results=[]
cases = [
 ('luna_top5', {'model':'gpt-6-luna','reasoning':{'effort':'none'},'include':['message.output_text.logprobs'],'top_logprobs':5}),
 ('luna_sentence', {'model':'gpt-6-luna','reasoning':{'effort':'none'},'include':['message.output_text.logprobs'],'input':[{'role':'user','content':[{'type':'input_text','text':'Write one short sentence about rain.'}]}]}),
 ('sol_none', {'model':'gpt-6-sol','reasoning':{'effort':'none'},'include':['message.output_text.logprobs']}),
 ('terra_none', {'model':'gpt-5.6-terra','reasoning':{'effort':'none'},'include':['message.output_text.logprobs']}),
 ('luna56_none', {'model':'gpt-5.6-luna','reasoning':{'effort':'none'},'include':['message.output_text.logprobs']}),
]
for name, extra in cases:
    body = dict(base, **extra)
    body = {k:v for k,v in body.items() if v is not None}
    req=urllib.request.Request('https://chatgpt.com/backend-api/codex/responses',
        data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req,timeout=45) as response:
            raw=response.read().decode()
            events=[]
            for line in raw.splitlines():
                if line.startswith('data: ') and line[6:]!='[DONE]':
                    try: events.append(json.loads(line[6:]))
                    except ValueError: pass
            item={'test':name,'status':response.status,'events':events}
    except urllib.error.HTTPError as e:
        item={'test':name,'status':e.code,'error':e.read().decode()[:2000]}
    except Exception as e:
        item={'test':name,'error':str(e)}
    serialized=json.dumps(item)
    for secret in tokens.values():
        if isinstance(secret,str) and secret: serialized=serialized.replace(secret,'[REDACTED]')
    item=json.loads(serialized)
    results.append(item)
    print(json.dumps({k:v for k,v in item.items() if k!='events'} | {'tokens':[e.get('logprobs') for e in item.get('events',[]) if e.get('type')=='response.output_text.done']}),flush=True)
pathlib.Path('pbgp-pilot/codex_followup_probe.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
