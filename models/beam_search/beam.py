from __future__ import division
import torch
from models.beam_search import penalties

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

class Beam(object):
    """
    Class for managing the internals of the beam search process.

    Takes care of beams, back pointers, and scores.

    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """

    def __init__(self, size, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0,
                 stepwise_penalty=False,
                 block_ngram_repeat=0,
                 exclusion_tokens=set()):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                        .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

        # Apply Penalty at every step
        self.stepwise_penalty = stepwise_penalty
        self.block_ngram_repeat = block_ngram_repeat
        self.exclusion_tokens = exclusion_tokens

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    def advance(self, word_probs, attn_out):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.

        Parameters:

        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step

        Returns: True if beam search is complete.
        """
        num_words = word_probs.size(1)
        if self.stepwise_penalty:
            self.global_scorer.update_score(self, attn_out)
        # force the output to be longer than self.min_length
        cur_len = len(self.next_ys)
        if cur_len < self.min_length:
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20
        # Sum the previous scores.
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + \
                self.scores.unsqueeze(1).expand_as(word_probs)
            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20

            # Block ngram repeats
            if self.block_ngram_repeat > 0:
                ngrams = []
                le = len(self.next_ys)
                for j in range(self.next_ys[-1].size(0)):
                    hyp, _ = self.get_hyp(le - 1, j)
                    ngrams = set()
                    fail = False
                    gram = []
                    for i in range(le - 1):
                        # Last n tokens, n = block_ngram_repeat
                        gram = (gram +
                                [hyp[i].item()])[-self.block_ngram_repeat:]
                        # Skip the blocking if it is in the exclusion list
                        if set(gram) & self.exclusion_tokens:
                            continue
                        if tuple(gram) in ngrams:
                            fail = True
                        ngrams.add(tuple(gram))
                    if fail:
                        beam_scores[j] = -10e20
        else:
            beam_scores = word_probs[0]
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0,
                                                            True, True)

        self.all_scores.append(self.scores)
        self.scores = best_scores

        # best_scores_id is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = best_scores_id / num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append((best_scores_id - prev_k * num_words))
        self.attn.append(attn_out.index_select(0, prev_k))
        self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                global_scores = self.global_scorer.score(self, self.scores)
                s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))
                i += 1

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1], torch.stack(attn[::-1])


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`

    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, alpha,   length_penalty):
        self.alpha = alpha
        penalty_builder = penalties.PenaltyBuilder(length_penalty)
        # Term will be subtracted from probability
        # Probability will be divided by this
        self.length_penalty = penalty_builder.length_penalty()

    def score(self, beam, logprobs):
        """
        Rescores a prediction based on penalty functions
        """
        normalized_probs = self.length_penalty(beam,
                                               logprobs,
                                               self.alpha)

        return normalized_probs


class BeamSearchDecoding(object):
    def __init__(self, model, generator, tokenizer, block_trigram, beam_size):

        self.model = model
        self.generator = generator
        self.tokenizer = tokenizer
        self.end_token_id = self.tokenizer.eos_token_id
        self.block_trigram = block_trigram
        self.beam_size = beam_size

    def run(self, src, src_features, mask_src, 
            tgt, prompts_indicator, 
            src_txts, tgt_txts,
            min_length, max_length, device, 
            example_ids, init_embeddings=None):

        beam_size = self.beam_size
        batch_size = src_features.size(0)

        # Tile states and memory beam_size times.
        mask_src = tile(mask_src, beam_size, dim=0)
        src_features = tile(src_features, beam_size, dim=0)

        batch_offset = torch.arange(batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(0, batch_size * beam_size, step=beam_size, dtype=torch.long, device=device)
        alive_seq = tile(tgt[:,0], beam_size, dim=0).unsqueeze(1)

        # Tile tgt and mask_tgt_for_loss if the input has prompts
        prompts = tile(tgt, beam_size, dim=0)
        prompts_indicator = tile(prompts_indicator, beam_size, dim=0)
        if init_embeddings is not None:
            init_embeddings = tile(init_embeddings, beam_size, dim=0)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (beam_size - 1), device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]
        results["scores"] = [[] for _ in range(batch_size)]
        results["src_txts"] = src_txts
        results["tgt_txts"] = tgt_txts
        results["src"] = src
        results["eid"] = example_ids

        for step in range(max_length):

            #decoder_input = alive_seq[:, -1].view(1, -1)
            decoder_input = alive_seq

            if init_embeddings is not None:
                tgt_embs = self.model.decoder.embed_tokens(decoder_input)
                if tgt_embs.size(1) == 1:
                    tgt_embs = init_embeddings
                else:
                    tgt_embs = torch.cat((init_embeddings, tgt_embs[:,1:,:]), dim=1)
                decoder_outputs = self.model.decoder(inputs_embeds=tgt_embs,
                                                     encoder_hidden_states=src_features,
                                                     encoder_attention_mask=mask_src)
            else:
                decoder_outputs = self.model.decoder(input_ids=decoder_input,
                                                     encoder_hidden_states=src_features,
                                                     encoder_attention_mask=mask_src)

            dec_out = decoder_outputs.last_hidden_state[:, -1, :]

            # Generator forward.
            log_probs = self.generator.forward(dec_out)
            vocab_size = log_probs.size(-1)

            #if step < min_length:
            #    print (prompts_indicator)
            for i in range(prompts_indicator.size(0)):
                if step - (~prompts_indicator[i]).sum() < min_length:
                    log_probs[i, self.end_token_id] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            #alpha = self.global_scorer.alpha
            #length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            length_penalty = 1.0

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if(self.block_trigram):
                cur_len = alive_seq.size(1)
                if(cur_len>3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = self.tokenizer.decode(words).split()
                        if(len(words)<=3):
                            continue
                        trigrams = [(words[i-1],words[i],words[i+1]) for i in range(1,len(words)-1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size).int()
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (topk_beam_index + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            cur_idx = step+1
            for ex_idx in range(alive_seq.size(0)):
                if cur_idx < prompts_indicator.size(1) and prompts_indicator[ex_idx][cur_idx] == False:
                    alive_seq[ex_idx][-1] = prompts[ex_idx][cur_idx]
                    if ex_idx % beam_size == 0:
                        topk_log_probs[int(ex_idx/beam_size)][(ex_idx%beam_size)] = 0.0
                    else:
                        topk_log_probs[int(ex_idx/beam_size)][(ex_idx%beam_size)] = float("-inf")

            is_finished = topk_ids.eq(self.end_token_id)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished).view(-1, alive_seq.size(-1))

            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            mask_src = mask_src.index_select(0, select_indices)
            prompts = prompts.index_select(0, select_indices)
            prompts_indicator = prompts_indicator.index_select(0, select_indices)
            if init_embeddings is not None:
                init_embeddings = init_embeddings.index_select(0, select_indices)

        return results


