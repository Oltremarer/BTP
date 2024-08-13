def get_predict_sequence(self, state, horizon=1024):
    # 初始状态
    beams = [(state, 0)]
    results = []

    for _ in range(horizon):
        new_beams = []
        for seq, score in beams:
            # 使用模型预测下一个 token
            input_ids = torch.tensor(seq).unsqueeze(0).to(self.device)
            outputs = self.model.generate(input_ids, max_length=len(seq) + 1, num_beams=self.num_beams,
                                          num_return_sequences=self.num_beams)
            for output in outputs:
                new_seq = output.tolist()
                new_score = score + self._score(new_seq)
                new_beams.append((new_seq, new_score))

        # 按分数排序并保留 top k
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:self.k]
        beams = new_beams
        results.extend(new_beams)

    # 返回分数最高的序列
    best_seq = max(results, key=lambda x: x[1])[0]
    return best_seq


def _score(self, seq):
    # 计算序列的分数
    return sum(self.model(input_ids=torch.tensor(seq).unsqueeze(0).to(self.device)).logits[0, -1, :])