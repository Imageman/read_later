from __future__ import annotations
from pathlib import Path

import gradio as gr
import pandas as pd
from loguru import logger

import rating_store


@logger.catch
def refresh_table():
    df = pd.DataFrame(rating_store.get_all_records())
    # show rating before text to keep numbers visible even for long texts
    return df[["id", "rating", "text"]]


@logger.catch
def add(text, rating):
    rating_store.add_record(text, float(rating))
    return refresh_table(), gr.update(choices=rating_store.get_all_ids())


@logger.catch
def update(record_id, text, rating):
    rating_store.update_record(record_id, text or None, float(rating) if rating is not None else None)
    return refresh_table(), gr.update(choices=rating_store.get_all_ids())


@logger.catch
def delete(record_id):
    rating_store.delete_record(record_id)
    return refresh_table(), gr.update(choices=rating_store.get_all_ids())


@logger.catch
def do_import(file):
    if file is not None:
        rating_store.import_json(file.name)
    return refresh_table(), gr.update(choices=rating_store.get_all_ids())


@logger.catch
def do_export():
    path = rating_store.DATA_DIR / "export.json"
    rating_store.export_json(path)
    return str(path)


@logger.catch
def do_recalculate():
    rating_store.recalculate_embeddings()
    return refresh_table(), gr.update(choices=rating_store.get_all_ids())


def predict(text, n, epsilon):
    return rating_store.predict_rating(text, int(n), float(epsilon))


def grid_search(n_values, epsilon_values):
    n_vals = [int(v) for v in n_values.split(",") if v.strip()]
    eps_vals = [float(v) for v in epsilon_values.split(",") if v.strip()]
    n, eps, rmse = rating_store.grid_search_n_epsilon(n_vals, eps_vals)
    return f"best_n={n}, best_epsilon={eps}, rmse={rmse:.4f}"


def build_demo():
    with gr.Blocks(css="""
        /* ТОЛЬКО наша таблица по elem_id */
        #ratings_table table th:nth-child(1),
        #ratings_table table td:nth-child(1) {
            min-width: 140px;        /* id */
        }
        #ratings_table table th:nth-child(2),
        #ratings_table table td:nth-child(2) {
            min-width: 50px;         /* rating */
            text-align: right;       /* опционально */
        }
        """) as demo:
        gr.Markdown("# Ratings storage")
        table = gr.DataFrame(headers=["id", "rating", "text"], interactive=False, wrap=True, max_height=500, elem_id="ratings_table")
        id_dd = gr.Dropdown(label="Record ID",  min_width=130)

        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(
            lambda: (refresh_table(), gr.update(choices=rating_store.get_all_ids())),
            outputs=[table, id_dd],
        )

        with gr.Row():
            text_in = gr.Textbox(label="Text")
            rating_in = gr.Slider(minimum=0, maximum=10, step=0.1, label="Rating")
            add_btn = gr.Button("Add")
        add_btn.click(add, inputs=[text_in, rating_in], outputs=[table, id_dd])

        with gr.Row():
            new_text = gr.Textbox(
                label="Updated Text",
                lines=2,
                placeholder="Leave blank to keep current",
            )
            new_rating = gr.Slider(minimum=0, maximum=10, step=0.1, label="Updated Rating")
            upd_btn = gr.Button("Update")
            del_btn = gr.Button("Delete")
        upd_btn.click(update, inputs=[id_dd, new_text, new_rating], outputs=[table, id_dd])
        del_btn.click(delete, inputs=id_dd, outputs=[table, id_dd])

        @logger.catch
        def load_record(record_id):
            rec = rating_store.get_record(record_id)
            if not rec:
                return "", 0.0
            return rec.get("text", ""), rec.get("rating", 0.0)

        id_dd.change(load_record, inputs=id_dd, outputs=[new_text, new_rating])

        import_file = gr.File(label="Import JSON")
        import_file.upload(do_import, inputs=import_file, outputs=[table, id_dd])
        export_btn = gr.Button("Export JSON")
        export_file = gr.File(label="Exported JSON")
        export_btn.click(do_export, outputs=export_file)

        recalc_btn = gr.Button("Recalculate Embeddings")
        recalc_btn.click(do_recalculate, outputs=[table, id_dd])

        with gr.Row():
            pred_text = gr.Textbox(label="Predict text")
            pred_n = gr.Number(label="n", value=3)
            pred_eps = gr.Number(label="epsilon", value=0.001)
            pred_btn = gr.Button("Predict Rating")
            pred_out = gr.Number(label="Predicted rating")
        pred_btn.click(predict, inputs=[pred_text, pred_n, pred_eps], outputs=pred_out)

        with gr.Row():
            n_vals = gr.Textbox(label="n values", value="1,3,5,7")
            eps_vals = gr.Textbox(label="epsilon values", value="0.001,0.01,0.1")
            grid_btn = gr.Button("Grid Search")
            grid_out = gr.Textbox(label="Grid Search Result")
        grid_btn.click(grid_search, inputs=[n_vals, eps_vals], outputs=grid_out)

        demo.load(
            lambda: (refresh_table(), gr.update(choices=rating_store.get_all_ids())),
            outputs=[table, id_dd],
        )
    return demo


if __name__ == "__main__":
    rating_store.init_db()
    build_demo().launch()
